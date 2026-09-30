"""可答性门禁：决定这次检索结果够不够回答问题，不够就拒答。

三种模式，都挂在同一个接口上，切换只改配置：
    off        不判定，检索到什么答什么（基线行为，用于对照）
    heuristic  关键词覆盖率，零成本、确定性（实测净收益为负，见 README）
    llm        LLM 判官，语义判断（正解）

判定结果会被记进 RunRecord.extra，badcase 复盘时能直接看到"为什么拒答/为什么敢答"。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .judge import LLMJudge
from .llm import ChatClient
from .relevance import query_coverage


@dataclass
class GateDecision:
    answerable: bool
    mode: str
    source: str
    reason: str = ""
    error: str | None = None
    coverage: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class OffGate:
    mode = "off"

    def check(self, query: str, contexts: list[str]) -> GateDecision:
        return GateDecision(answerable=True, mode=self.mode, source="off", reason="门禁未启用")


class HeuristicGate:
    mode = "heuristic"

    def __init__(self, min_query_coverage: float = 0.0):
        self.min_query_coverage = float(min_query_coverage)

    def check(self, query: str, contexts: list[str]) -> GateDecision:
        if not contexts:
            return GateDecision(answerable=False, mode=self.mode, source="heuristic", reason="没有检索到任何资料", coverage=0.0)
        coverage = query_coverage(query, contexts)
        answerable = coverage >= self.min_query_coverage
        return GateDecision(
            answerable=answerable,
            mode=self.mode,
            source="heuristic",
            reason=f"关键词覆盖率 {coverage:.2f} {'≥' if answerable else '<'} 阈值 {self.min_query_coverage:.2f}",
            coverage=round(coverage, 4),
        )


class LLMGate:
    mode = "llm"

    def __init__(self, judge: LLMJudge):
        self.judge = judge

    def check(self, query: str, contexts: list[str]) -> GateDecision:
        if not contexts:
            return GateDecision(answerable=False, mode=self.mode, source="llm", reason="没有检索到任何资料")
        result = self.judge.judge_answerability(query, contexts)
        return GateDecision(
            answerable=result.answerable,
            mode=self.mode,
            source=result.source,
            reason=result.reason,
            error=result.error,
        )


def build_gate(cfg):
    raw_mode = cfg.get("answerability.mode", "off")
    # YAML 会把裸 off/on 解析成布尔，这里统一归一化
    if raw_mode is False:
        raw_mode = "off"
    elif raw_mode is True:
        raw_mode = "on"
    mode = str(raw_mode).lower()
    if mode == "off":
        return OffGate()
    if mode == "heuristic":
        return HeuristicGate(min_query_coverage=float(cfg.get("answerability.min_query_coverage", 0.0)))
    if mode == "llm":
        cache_dir = cfg.path_of("judge.cache_dir") if cfg.get("judge.cache_dir") else None
        client = ChatClient(
            provider=str(cfg.get("judge.provider", "deepseek")),
            model=cfg.get("judge.model"),
            timeout=float(cfg.get("judge.timeout_seconds", 60)),
        )
        judge = LLMJudge(
            client=client,
            cache_dir=Path(cache_dir) if cache_dir else None,
            cache_only=bool(cfg.get("judge.cache_only", False)),
            on_error=str(cfg.get("answerability.on_error", "fallback")),
        )
        return LLMGate(judge)
    raise ValueError(f"未支持的 answerability.mode：{mode}")
