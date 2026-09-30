"""LLM 判官：可答性判定 + 原子 claim 拆解。

为什么需要它：关键词覆盖率在 golden 集上被实测否决（净收益为负，见 README）。
词汇级信号分不清"问法和原文用词不同"与"知识库没有这块知识"，语义判官可以。

两个工程约束：
1. **缓存**：判官结果按 (模型, 问句, 资料) 哈希落盘复用。同一批数据反复跑不重复花钱。
2. **降级**：无外网的机器上判官必然调不通。连续失败若干次后触发熔断，
   整批不再尝试，按配置的 on_error 策略兜底，绝不让跑批卡死。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .llm import ChatClient, LLMError

JUDGE_VERSION = "judge-v1"

ANSWERABILITY_PROMPT = """你是知识库问答系统的质检员。请判断下面的【资料】是否足以回答【问题】。

判定标准：
1. 只要资料里存在能支撑答案的事实依据，就判"可回答"。即使问法口语化、用词与资料不同（同义改写），也算可回答。
2. 资料完全没有涉及问题所问的信息（知识库没收录这个主题），判"不可回答"。
3. 资料只是提到相关主题，但没有回答问题所问的具体信息，判"不可回答"。

【资料】
{context}

【问题】
{query}

只输出一个 JSON 对象，不要输出任何其他文字或代码块标记：
{{"answerable": true, "reason": "一句话说明判断依据"}}
"""

CLAIM_PROMPT = """你是知识库问答系统的质检员。请把【回答】拆成最小的原子事实（claim），
逐条判断它能否被【资料】直接支持。

规则：
1. 拆分到"一个 claim 只包含一个事实"为止，不要把多个事实合并。
2. supported 为 true 表示资料里有明确依据；资料没提到、或与资料矛盾，都为 false。
3. evidence 填资料中支持该 claim 的原文片段；不支持则填空字符串。

【资料】
{context}

【问题】
{query}

【回答】
{answer}

只输出一个 JSON 对象，不要输出任何其他文字或代码块标记：
{{"claims": [{{"claim": "原子事实", "supported": true, "evidence": "资料原文片段"}}]}}
"""

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    """模型有时会包 ```json 或加前后缀，这里做一次宽松提取。"""
    match = _JSON_BLOCK.search(text)
    if not match:
        raise ValueError(f"判官返回的不是 JSON：{text[:200]}")
    return json.loads(match.group(0))


def render_context(contexts: list[str], max_chars: int = 3000) -> str:
    lines: list[str] = []
    used = 0
    for index, text in enumerate(contexts, start=1):
        text = text.strip()
        if used + len(text) > max_chars:
            text = text[: max(0, max_chars - used)].rstrip()
        if not text:
            break
        lines.append(f"[{index}] {text}")
        used += len(text)
    return "\n\n".join(lines)


@dataclass
class JudgeResult:
    answerable: bool
    reason: str = ""
    source: str = "llm"        # llm | cache | fallback
    error: str | None = None
    raw: dict = field(default_factory=dict)


class LLMJudge:
    """带缓存与熔断的判官。"""

    def __init__(
        self,
        client: ChatClient,
        cache_dir: Path | None = None,
        cache_only: bool = False,
        on_error: str = "fallback",       # fallback（按可答处理=基线行为） | refuse | raise
        failure_circuit: int = 3,
        max_context_chars: int = 3000,
    ):
        self.client = client
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_only = cache_only
        self.on_error = on_error
        self.failure_circuit = int(failure_circuit)
        self.max_context_chars = int(max_context_chars)
        self._consecutive_failures = 0
        self._tripped = False
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- 缓存 ----

    def _cache_file(self, prompt: str) -> Path | None:
        if not self.cache_dir:
            return None
        digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:20]
        return self.cache_dir / f"{digest}.json"

    def _cache_read(self, prompt: str) -> dict | None:
        path = self._cache_file(prompt)
        if path and path.exists():
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        return None

    def _cache_write(self, prompt: str, payload: dict) -> None:
        path = self._cache_file(prompt)
        if path:
            with path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)

    # ---- 对外接口 ----

    def _call_cached(self, prompt: str) -> tuple[dict | None, str | None, str]:
        """返回 (payload, error, source)。source 取 llm / cache / fallback。"""
        cached = self._cache_read(prompt)
        if cached is not None:
            return cached, None, "cache"
        if self.cache_only:
            return None, "cache_only 模式下缓存未命中", "fallback"
        if self._tripped:
            return None, f"判官已熔断（连续失败 {self._consecutive_failures} 次）", "fallback"
        try:
            result = self.client.chat(prompt)
            self._consecutive_failures = 0
            payload = _extract_json(result.text)
            self._cache_write(prompt, payload)
            return payload, None, "llm"
        except (LLMError, ValueError, KeyError) as exc:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_circuit:
                self._tripped = True
            return None, str(exc), "fallback"

    def _degraded(self, error: str) -> JudgeResult:
        if self.on_error == "raise":
            raise LLMError(error)
        answerable = self.on_error != "refuse"
        return JudgeResult(answerable=answerable, reason="判官不可用，按兜底策略处理", source="fallback", error=error)

    def judge_answerability(self, query: str, contexts: list[str]) -> JudgeResult:
        prompt = ANSWERABILITY_PROMPT.format(context=render_context(contexts, self.max_context_chars), query=query)
        payload, error, source = self._call_cached(prompt)
        if payload is None:
            return self._degraded(error or "未知错误")
        return JudgeResult(
            answerable=bool(payload.get("answerable", True)),
            reason=str(payload.get("reason", "")),
            source=source,
            raw=payload,
        )

    def judge_claims(self, query: str, answer: str, contexts: list[str]) -> dict:
        """原子 claim 拆解。供忠实度指标使用。"""
        prompt = CLAIM_PROMPT.format(
            context=render_context(contexts, self.max_context_chars), query=query, answer=answer
        )
        payload, error, _source = self._call_cached(prompt)
        if payload is None:
            return {"claims": [], "faithfulness": None, "error": error}
        claims = payload.get("claims", [])
        supported = [c for c in claims if c.get("supported")]
        return {
            "claims": claims,
            "faithfulness": round(len(supported) / len(claims), 4) if claims else None,
            "error": None,
        }
