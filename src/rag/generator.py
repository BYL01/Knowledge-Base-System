"""生成层。可以脱离检索层单独调用（喂假上下文即可），忠实度和安全指标在这里测。"""

from __future__ import annotations

import os
import re

from .embedder import tokenize
from .schema import ScoredChunk

PROMPT_V0 = """你是「电商知识库」的客服问答助手，面向平台用户答题。

【回答要求】
1. 只能依据下面给出的【资料】回答，资料里没有的信息不要编造，也不要凭常识补充。
2. 每个结论后面必须标注来源编号，格式形如 [1]。
3. 如果资料不足以回答该问题，直接回答"知识库中没有找到相关内容"。

【资料】
{context_block}

【问题】
{query}

【回答】
"""

PROMPT_TEMPLATES = {"v0": PROMPT_V0}


def render_contexts(contexts: list[ScoredChunk], max_chars: int = 2400) -> str:
    lines: list[str] = []
    used = 0
    for index, item in enumerate(contexts, start=1):
        text = item.chunk.text.strip()
        if used + len(text) > max_chars:
            text = text[: max(0, max_chars - used)].rstrip()
        if not text:
            break
        lines.append(f"[{index}] 来源：{item.chunk.source}\n{text}")
        used += len(text)
    return "\n\n".join(lines)


def render_prompt(query: str, contexts: list[ScoredChunk], prompt_version: str = "v0", max_chars: int = 2400) -> str:
    template = PROMPT_TEMPLATES.get(prompt_version)
    if template is None:
        raise ValueError(f"未支持的 prompt_version：{prompt_version}")
    return template.format(context_block=render_contexts(contexts, max_chars), query=query)


def _split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for paragraph in re.split(r"(?<=[。！？!?；;])", text):
        for line in paragraph.splitlines():
            line = line.strip().lstrip("#").strip()
            if line:
                sentences.append(line)
    return sentences


def _overlap_score(query_tokens: set[str], sentence: str) -> float:
    sentence_tokens = set(tokenize(sentence))
    if not sentence_tokens:
        return 0.0
    common = query_tokens & sentence_tokens
    return len(common) / (len(query_tokens) ** 0.5 * len(sentence_tokens) ** 0.5)


class ExtractiveGenerator:
    """抽取式基线：从命中的 chunk 里挑最贴题的原句拼接，并强制带引用。

    它的作用不是"答得好"，而是让链路和评测口径先稳定下来：
    确定性输出 = CI 可复现 = 换模型时能对比出真实差异。
    """

    name = "extractive"

    def __init__(
        self,
        prompt_version: str = "v0",
        max_context_chars: int = 2400,
        max_citations: int = 2,
    ):
        self.prompt_version = prompt_version
        self.max_context_chars = int(max_context_chars)
        self.max_citations = int(max_citations)

    def build_prompt(self, query: str, contexts: list[ScoredChunk]) -> str:
        return render_prompt(query, contexts, self.prompt_version, self.max_context_chars)

    def generate(self, query: str, contexts: list[ScoredChunk], prompt: str | None = None) -> tuple[str, list[str]]:
        if not contexts:
            return "知识库中没有找到相关内容", []

        query_tokens = set(tokenize(query))
        scored: list[tuple[float, int, str]] = []
        for index, item in enumerate(contexts):
            sentences = _split_sentences(item.chunk.text)
            if not sentences:
                continue
            best = max(sentences, key=lambda s: _overlap_score(query_tokens, s))
            scored.append((_overlap_score(query_tokens, best), index, best))
        if not scored:
            return "知识库中没有找到相关内容", []

        # 先按贴合度取前 N 条，再恢复成"资料编号顺序"，读起来更自然
        picked = sorted(scored, key=lambda row: (-row[0], row[1]))[: self.max_citations]
        picked.sort(key=lambda row: row[1])

        pieces: list[str] = []
        citations: list[str] = []
        for _, index, sentence in picked:
            pieces.append(f"{sentence}[{index + 1}]")
            citations.append(contexts[index].chunk.chunk_id)
        return "根据知识库资料：" + " ".join(pieces), citations


class OpenAIGenerator:
    """真实模型生成。需要 OPENAI_API_KEY。"""

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        prompt_version: str = "v0",
        max_context_chars: int = 2400,
        temperature: float = 0.0,
        base_url: str | None = None,
    ):
        import requests

        self._requests = requests
        self.model = model
        self.prompt_version = prompt_version
        self.max_context_chars = int(max_context_chars)
        self.temperature = float(temperature)
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("generator.backend=openai 需要先设置 OPENAI_API_KEY")

    def build_prompt(self, query: str, contexts: list[ScoredChunk]) -> str:
        return render_prompt(query, contexts, self.prompt_version, self.max_context_chars)

    def generate(self, query: str, contexts: list[ScoredChunk], prompt: str | None = None) -> tuple[str, list[str]]:
        prompt = prompt or self.build_prompt(query, contexts)
        response = self._requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120,
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"].strip()
        citations = [
            contexts[int(number) - 1].chunk.chunk_id
            for number in dict.fromkeys(re.findall(r"\[(\d+)\]", answer))
            if 1 <= int(number) <= len(contexts)
        ]
        return answer, citations


def build_generator(cfg):
    backend = str(cfg.get("generator.backend", "extractive")).lower()
    common = {
        "prompt_version": str(cfg.get("generator.prompt_version", "v0")),
        "max_context_chars": int(cfg.get("generator.max_context_chars", 2400)),
    }
    if backend == "extractive":
        return ExtractiveGenerator(max_citations=int(cfg.get("generator.max_citations", 2)), **common)
    if backend == "openai":
        return OpenAIGenerator(
            model=str(cfg.get("generator.model", "gpt-4o-mini")),
            temperature=float(cfg.get("generator.temperature", 0.0)),
            base_url=cfg.env("OPENAI_BASE_URL"),
            **common,
        )
    raise ValueError(f"未支持的 generator.backend：{backend}")
