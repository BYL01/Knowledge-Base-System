"""跑批：把 golden 集跑一遍，落盘中间产物 + 打印指标。

用法：
    python scripts/run_batch.py
    python scripts/run_batch.py --golden data/golden/golden_v0.jsonl --limit 5

产物：
    outputs/runs/<run_id>.jsonl   每次问答的完整中间产物（含 prompt、retrieved、answer）
    outputs/runs/<run_id>.summary.json  本次跑批的指标汇总
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:  # Windows 控制台默认 cp936，中文输出会乱码
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:  # pragma: no cover
    pass

from src.rag.config import load_config, load_env_file  # noqa: E402
from src.rag.pipeline import RAGPipeline  # noqa: E402

REFUSAL_MARKERS = ("没有找到相关内容", "没有找到", "无法回答", "未收录")


def load_golden(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"golden 集不存在：{path}")
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def evaluate(records: list[dict], golden: list[dict]) -> dict:
    golden_by_id = {g["id"]: g for g in golden}
    n = len(records)
    if n == 0:
        return {"count": 0}

    doc_hits: list[float] = []
    chunk_hits: list[float] = []
    chunk_recalls: list[float] = []
    refusal_correct: list[float] = []
    false_refusals: list[float] = []
    dangling_total = 0

    for record in records:
        gold = golden_by_id.get(record.get("golden_id") or "", {})
        retrieved_ids = [item["chunk_id"] for item in record["retrieved"]]
        retrieved_sources = {item["source"] for item in record["retrieved"]}
        gold_sources = set(gold.get("gold_sources") or [])
        gold_chunks = set(gold.get("gold_chunk_ids") or [])

        is_refusal = gold.get("expect_refusal", False)
        answered_refusal = any(marker in record["answer"] for marker in REFUSAL_MARKERS)

        if is_refusal:
            # 无答案题：应当拒答；如果还硬检索出内容并作答，算失败
            refusal_correct.append(1.0 if answered_refusal else 0.0)
        else:
            if gold_sources:
                doc_hits.append(1.0 if gold_sources & retrieved_sources else 0.0)
            if gold_chunks:
                chunk_hits.append(1.0 if gold_chunks & set(retrieved_ids) else 0.0)
                chunk_recalls.append(len(gold_chunks & set(retrieved_ids)) / len(gold_chunks))
            # 有答案的题不该被拒答
            false_refusals.append(1.0 if answered_refusal else 0.0)

        dangling_total += len(record.get("extra", {}).get("dangling_citations", []))

    latencies = [record["latency_ms"] for record in records]
    return {
        "count": n,
        "doc_hit_rate": round(statistics.fmean(doc_hits), 4) if doc_hits else None,
        "chunk_hit_rate": round(statistics.fmean(chunk_hits), 4) if chunk_hits else None,
        "chunk_recall_at_k": round(statistics.fmean(chunk_recalls), 4) if chunk_recalls else None,
        "refusal_accuracy": round(statistics.fmean(refusal_correct), 4) if refusal_correct else None,
        "false_refusal_rate": round(statistics.fmean(false_refusals), 4) if false_refusals else None,
        "dangling_citations": dangling_total,
        "avg_latency_ms": round(statistics.fmean(latencies), 1),
        "p95_latency_ms": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="跑 golden 集并落盘中间产物")
    parser.add_argument("--config", default=None)
    parser.add_argument("--golden", default=None, help="默认取配置里的 golden.path")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，0 表示全跑")
    args = parser.parse_args()

    load_env_file()
    cfg = load_config(args.config)
    golden_path = Path(args.golden) if args.golden else cfg.path_of("golden.path")
    if not golden_path.is_absolute():
        golden_path = Path.cwd() / golden_path

    golden = load_golden(golden_path)
    if args.limit:
        golden = golden[: args.limit]

    pipeline = RAGPipeline.from_config(cfg)
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    output_dir = cfg.path_of("run.output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{run_id}.jsonl"

    records: list[dict] = []
    with output_file.open("w", encoding="utf-8") as fh:
        for item in golden:
            record = pipeline.answer(item["query"], golden_id=item["id"], category=item.get("category"))
            payload = record.to_dict()
            records.append(payload)
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            print(f"[{item['id']}] {item['query']}\n    -> {record.answer}")

    summary = {
        "run_id": run_id,
        "golden_file": str(golden_path),
        "config": str(cfg.path),
        "index_version": records[0]["index_version"] if records else None,
        "prompt_version": records[0]["prompt_version"] if records else None,
        "embedding_backend": records[0]["embedding_backend"] if records else None,
        "generator_backend": records[0]["generator_backend"] if records else None,
        "metrics": evaluate(records, golden),
    }
    summary_file = output_dir / f"{run_id}.summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n==== 指标 ====")
    for key, value in summary["metrics"].items():
        print(f"{key:>28}: {value}")
    print(f"\n中间产物: {output_file}")
    print(f"指标汇总: {summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
