"""冒烟测试：保证链路端到端可用，且分层边界没被打破。

这些用例跑得快、结果确定，可以直接当成 CI 的第一道门禁。
"""

from __future__ import annotations

import json

import pytest

from src.rag.config import PROJECT_ROOT, load_config
from src.rag.ingest import build_index
from src.rag.pipeline import RAGPipeline
from src.rag.relevance import query_coverage
from src.rag.retriever import build_retriever


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    """索引写到临时目录，不污染仓库里的 data/index。

    同时把可答性门禁强制关掉：冒烟测试要的是确定性和零外部依赖，
    不能因为 configs 里切成 llm 就去真调模型。
    """
    config = load_config()
    index_dir = tmp_path_factory.mktemp("index")
    config.set("store.path", str(index_dir))
    config.set("answerability.mode", "off")
    build_index(config, verbose=False)
    return config


@pytest.fixture(scope="module")
def pipeline(cfg):
    return RAGPipeline.from_config(cfg)


def test_retriever_layer_is_independent(cfg):
    """检索层必须能脱离生成层单独调用——检索指标就挂在这里。"""
    retriever = build_retriever(cfg)
    hits = retriever.retrieve("无理由退货是几天")
    assert hits, "检索结果不应为空"
    assert all(hit.chunk.text.strip() for hit in hits)
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True), "检索结果必须按分数降序"


def test_retrieval_hits_expected_document(cfg):
    retriever = build_retriever(cfg)
    hits = retriever.retrieve("增值税专用发票审核通过后多久寄出")
    sources = [hit.chunk.source for hit in hits]
    assert "发票与开票.md" in sources


def test_answer_carries_traceable_citations(pipeline):
    record = pipeline.answer("退货申请通过后多久必须寄回商品？", golden_id="g0003")
    assert record.answer.strip()
    assert record.citations, "答案必须带引用"
    assert set(record.citations) <= set(record.retrieved_ids()), "引用必须能回溯到本次检索结果"
    assert not record.extra["dangling_citations"]
    assert "[1]" in record.prompt
    assert record.query in record.prompt


def test_record_is_jsonl_serializable(pipeline):
    record = pipeline.answer("积分有效期是多久？")
    line = json.dumps(record.to_dict(), ensure_ascii=False)
    restored = json.loads(line)
    assert restored["prompt_version"] == "v0"
    assert restored["index_version"] != "unknown"
    assert restored["retrieved"]


def test_golden_dataset_schema():
    golden_file = PROJECT_ROOT / "data" / "golden" / "golden_v1.jsonl"
    rows = [json.loads(line) for line in golden_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 10
    for row in rows:
        assert {"id", "query", "category", "gold_answer", "gold_sources"} <= set(row)
        if not row["expect_refusal"]:
            assert row["gold_sources"], f"{row['id']} 有答案的题必须标注来源文档"
            # 标注不能留空：gold_chunk_ids 由 scripts/resolve_evidence.py 按原文对齐
            assert row["gold_chunk_ids"], f"{row['id']} 未对齐到 chunk，请跑 resolve_evidence.py"


def test_query_coverage_signal():
    """覆盖率信号本身要能用：问句关键词在资料里出现过才算可答。"""
    context = ["本商城自营商品支持签收后 7 天内无理由退货，生鲜食品不支持无理由退货。"]
    assert query_coverage("生鲜食品可以无理由退货吗", context) > 0.6
    assert query_coverage("可以用花呗分期支付吗", context) < 0.4
