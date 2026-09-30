"""可答性判定（拒答门禁）。

为什么不用检索分数做阈值：基线实测下来，可答题的最高分最低到 0.10，
无答案题的最高分却到 0.28，两个区间完全重叠。任何绝对阈值都会同时误杀和漏放。

所以改用"查询关键词覆盖"：如果检索回来的内容里压根没出现过问句的关键词，
说明知识库没有这块知识，应当拒答。这个信号是确定性的、可解释的、零成本。

后面接 LLM-as-judge 时，这里是同一个挂载点：把 judge 结果替换掉这个函数即可。
"""

from __future__ import annotations

from .embedder import tokenize


def content_units(text: str) -> set[str]:
    """抽取有区分度的内容单元：中文二元组 + 长度≥2 的英文数字词。

    丢掉中文单字，因为"的、了、吗"这类单字没有区分度，会把覆盖率算虚高。
    """
    return {token for token in tokenize(text) if len(token) >= 2}


def query_coverage(query: str, contexts: list[str]) -> float:
    """问句关键内容在检索结果中的覆盖比例，取值 0~1。"""
    query_units = content_units(query)
    if not query_units:
        return 1.0
    context_units: set[str] = set()
    for text in contexts:
        context_units |= content_units(text)
    return len(query_units & context_units) / len(query_units)


def is_answerable(query: str, contexts: list[str], min_coverage: float) -> bool:
    if not contexts:
        return False
    return query_coverage(query, contexts) >= min_coverage
