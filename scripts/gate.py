"""质量门禁：指标不达标就返回非 0，供 CI 直接当卡口用。

用法：
    python scripts/gate.py                        # 校验最新一次跑批
    python scripts/gate.py --summary outputs/runs/xxx.summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:  # pragma: no cover
    pass

from src.rag.config import load_config  # noqa: E402

# 配置项 -> (指标名, 是否越大越好)
CHECKS = [
    ("min_doc_hit_rate", "doc_hit_rate", True),
    ("min_chunk_hit_rate", "chunk_hit_rate", True),
    ("min_refusal_accuracy", "refusal_accuracy", True),
    ("max_false_refusal_rate", "false_refusal_rate", False),
    ("max_dangling_citations", "dangling_citations", False),
    ("max_p95_latency_ms", "p95_latency_ms", False),
]


def latest_summary(output_dir: Path) -> Path:
    summaries = sorted(output_dir.glob("*.summary.json"))
    if not summaries:
        raise SystemExit("没有找到跑批汇总，先跑 python scripts/run_batch.py")
    return summaries[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="质量门禁校验")
    parser.add_argument("--config", default=None)
    parser.add_argument("--summary", default=None, help="跑批汇总文件，默认取最新一份")
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary_path = Path(args.summary) if args.summary else latest_summary(cfg.path_of("run.output_dir"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary.get("metrics", {})

    print(f"校验文件：{summary_path.name}  (index {summary.get('index_version')})")
    print(f"{'指标':<32}{'实际':>10}{'阈值':>10}   结果")

    failures: list[str] = []
    for config_key, metric_key, higher_is_better in CHECKS:
        threshold = cfg.get(f"gate.{config_key}")
        if threshold is None:
            continue
        actual = metrics.get(metric_key)
        if actual is None:
            print(f"{metric_key:<32}{'N/A':>10}{threshold:>10}   SKIP")
            continue
        passed = actual >= threshold if higher_is_better else actual <= threshold
        print(f"{metric_key:<32}{actual:>10}{threshold:>10}   {'PASS' if passed else 'FAIL'}")
        if not passed:
            failures.append(f"{metric_key}={actual} 未达到 {threshold}")

    if failures:
        print("\n门禁未通过：")
        for failure in failures:
            print(f"  ! {failure}")
        return 1
    print("\n门禁通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
