"""检索层。可以脱离生成层单独调用，检索指标就在这里测。"""

from __future__ import annotations

from .embedder import build_embedder
from .schema import ScoredChunk
from .store import build_store


class Retriever:
    def __init__(self, store, embedder, top_k: int = 4, min_score: float = 0.0):
        self.store = store
        self.embedder = embedder
        self.top_k = int(top_k)
        self.min_score = float(min_score)

    def retrieve(self, query: str) -> list[ScoredChunk]:
        vector = self.embedder.encode([query])[0]
        hits = self.store.search(vector, self.top_k) if hasattr(self.store, "search") else []
        return [ScoredChunk(chunk=chunk, score=score) for chunk, score in hits if score >= self.min_score]

    def index_version(self) -> str:
        try:
            _, _, manifest = self.store.load()
            return str(manifest.get("index_version", "unknown"))
        except NotImplementedError:
            return "unknown"


def build_retriever(cfg) -> Retriever:
    return Retriever(
        store=build_store(cfg),
        embedder=build_embedder(cfg),
        top_k=int(cfg.get("retriever.top_k", 4)),
        min_score=float(cfg.get("retriever.min_score", 0.0)),
    )
