"""编排层：把一次问答的全部中间产物封成 RunRecord 落盘。

评测、badcase 回流、CI 门禁都只消费 RunRecord，不重新跑链路。
"""

from __future__ import annotations

import time
import uuid

from .config import Config
from .answerability import build_gate
from .generator import build_generator
from .retriever import build_retriever
from .schema import RunRecord


REFUSAL_ANSWER = "知识库中没有找到相关内容"


class RAGPipeline:
    def __init__(self, retriever, generator, gate=None, index_version: str = "unknown"):
        self.retriever = retriever
        self.generator = generator
        self.gate = gate
        self._index_version = index_version

    @classmethod
    def from_config(cls, cfg: Config) -> "RAGPipeline":
        retriever = build_retriever(cfg)
        return cls(
            retriever=retriever,
            generator=build_generator(cfg),
            gate=build_gate(cfg),
            index_version=retriever.index_version(),
        )

    def retrieve(self, query: str):
        return self.retriever.retrieve(query)

    def answer(self, query: str, golden_id: str | None = None, category: str | None = None) -> RunRecord:
        started = time.perf_counter()

        contexts = self.retriever.retrieve(query)
        decision = self.gate.check(query, [item.chunk.text for item in contexts]) if self.gate else None
        if decision is not None and not decision.answerable:
            prompt, answer, citations = "", REFUSAL_ANSWER, []
        else:
            prompt = self.generator.build_prompt(query, contexts)
            answer, citations = self.generator.generate(query, contexts, prompt)

        latency_ms = int((time.perf_counter() - started) * 1000)

        # 引用必须能回溯到本次检索结果，回溯不了就是幻觉引用，这里直接暴露出来
        retrieved_ids = {item.chunk.chunk_id for item in contexts}
        dangling = [cid for cid in citations if cid not in retrieved_ids]

        return RunRecord(
            run_id=uuid.uuid4().hex[:12],
            query=query,
            retrieved=[item.to_dict() for item in contexts],
            prompt=prompt,
            answer=answer,
            citations=citations,
            latency_ms=latency_ms,
            index_version=self._index_version,
            prompt_version=str(getattr(self.generator, "prompt_version", "unknown")),
            embedding_backend=str(getattr(getattr(self.retriever, "embedder", None), "name", "unknown")),
            generator_backend=str(getattr(self.generator, "name", "unknown")),
            golden_id=golden_id,
            category=category,
            extra={
                "dangling_citations": dangling,
                "answerability": decision.to_dict() if decision else None,
            },
        )
