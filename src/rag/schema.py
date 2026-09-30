"""数据契约。评测代码只认这几个结构，换向量库/换模型都不影响。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Chunk:
    """检索的最小单元，必须能回溯到原文。"""

    chunk_id: str
    doc_id: str
    source: str
    text: str
    position: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "source": self.chunk.source,
            "position": self.chunk.position,
            "score": round(float(self.score), 6),
            "text": self.chunk.text,
        }


@dataclass
class RunRecord:
    """一次问答的全部中间产物。评测层吃这份数据，不用再回查链路。"""

    run_id: str
    query: str
    retrieved: list[dict[str, Any]]
    prompt: str
    answer: str
    citations: list[str]
    latency_ms: int
    index_version: str
    prompt_version: str
    embedding_backend: str
    generator_backend: str
    golden_id: str | None = None
    category: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def retrieved_ids(self) -> list[str]:
        return [item["chunk_id"] for item in self.retrieved]
