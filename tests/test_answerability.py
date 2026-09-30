"""可答性门禁的行为测试：三种模式 + 与 pipeline 的衔接。"""

from __future__ import annotations

import pytest

from src.rag.answerability import HeuristicGate, OffGate
from src.rag.pipeline import REFUSAL_ANSWER, RAGPipeline
from src.rag.schema import Chunk, ScoredChunk


class StubRetriever:
    def __init__(self, chunk: Chunk):
        self.chunk = chunk

    def retrieve(self, query):
        return [ScoredChunk(chunk=self.chunk, score=0.3)]

    def index_version(self):
        return "stub-index"


class StubGenerator:
    name = "stub"
    prompt_version = "v0"

    def build_prompt(self, query, contexts):
        return f"PROMPT::{query}"

    def generate(self, query, contexts, prompt=None):
        return f"ANSWER::{query}", [contexts[0].chunk.chunk_id]


@pytest.fixture
def chunk():
    return Chunk(
        chunk_id="退换货政策#00",
        doc_id="退换货政策",
        source="退换货政策.md",
        text="本商城自营商品支持签收后 7 天内无理由退货。生鲜食品等特殊商品不支持无理由退货。",
        position=0,
    )


def test_off_gate_always_answers(chunk):
    pipeline = RAGPipeline(StubRetriever(chunk), StubGenerator(), gate=OffGate())
    record = pipeline.answer("可以用花呗分期支付吗")
    assert record.answer.startswith("ANSWER::")
    assert record.extra["answerability"]["mode"] == "off"


def test_heuristic_gate_refuses_out_of_scope(chunk):
    pipeline = RAGPipeline(StubRetriever(chunk), StubGenerator(), gate=HeuristicGate(min_query_coverage=0.6))
    record = pipeline.answer("可以用花呗分期支付吗")
    assert record.answer == REFUSAL_ANSWER
    assert record.citations == []
    assert record.prompt == ""
    assert record.extra["answerability"]["answerable"] is False
    assert record.extra["answerability"]["coverage"] < 0.6


def test_heuristic_gate_allows_in_scope(chunk):
    pipeline = RAGPipeline(StubRetriever(chunk), StubGenerator(), gate=HeuristicGate(min_query_coverage=0.4))
    record = pipeline.answer("生鲜食品可以无理由退货吗")
    assert record.answer.startswith("ANSWER::")
    assert record.extra["answerability"]["answerable"] is True


def test_refusal_is_recorded_with_reason(chunk):
    """拒答必须留下理由，badcase 复盘时要能看出为什么拒。"""
    pipeline = RAGPipeline(StubRetriever(chunk), StubGenerator(), gate=HeuristicGate(min_query_coverage=0.9))
    record = pipeline.answer("可以用花呗分期支付吗")
    assert record.extra["answerability"]["reason"]
