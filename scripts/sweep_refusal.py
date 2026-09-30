"""扫描拒答门禁阈值，输出"拦住多少拒答题 / 误拒多少该答题"的权衡表。

阈值不能拍脑袋定。这个脚本把权衡关系摆出来，让决策有依据，也让结论可复现。

用法：
    python scripts/sweep_refusal.py                       # 用最新的跑批结果
    python scripts/sweep_refusal.py --run outputs/runs/xxx.jsonl
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

from src.rag.config import PROJECT_ROOT, load_config  # noqa: E402
from src.rag.relevance import query_coverage  # noqa: E402


def latest_run(output_dir: Path) -> Path:
    runs = sorted(output_dir.glob("*.jsonl"))
    if not runs:
        raise SystemExit("没有找到跑批结果，先跑 python scripts/run_batch.py")
    return runs[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="扫描拒答门禁阈值")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run", default=None, help="跑批产物 jsonl，默认取最新一份")
    parser.add_argument("--golden", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_path = Path(args.run) if args.run else latest_run(cfg.path_of("run.output_dir"))
    golden_path = Path(args.golden) if args.golden else cfg.path_of("golden.path")
    if not golden_path.is_absolute():
        golden_path = PROJECT_ROOT / golden_path

    gold = {json.loads(line)["id"]: json.loads(line) for line in golden_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    records = [json.loads(line) for line in run_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    answerable: list[tuple[float, str, str]] = []
    refusals: list[tuple[float, str, str]] = []
    for record in records:
        item = gold.get(record.get("golden_id") or "")
        if not item:
            continue
        coverage = query_coverage(record["query"], [c["text"] for c in record["retrieved"]])
        target = refusals if item["expect_refusal"] else answerable
        target.append((coverage, record["golden_id"], record["query"]))

    total = len(answerable) + len(refusals)
    print(f"跑批文件：{run_path.name}")
    print(f"可答题 {len(answerable)} 条，无答案题 {len(refusals)} 条\n")
    print("  阈值  |  拒绝正确  |  误拒答  |  综合准确率")
    print("  -------+------------+----------+-------------")
    for step in range(0, 21):
        threshold = step / 20
        refused_ok = sum(1 for c, _, _ in refusals if c < threshold)
        false_refuse = sum(1 for c, _, _ in answerable if c < threshold)
        accuracy = (len(answerable) - false_refuse + refused_ok) / total
        print(f"  {threshold:5.2f}  |   {refused_ok}/{len(refusals)}    |   {false_refuse:>3}    |   {accuracy:.4f}")

    print("\n可答题中覆盖率最低的 5 条（门禁最容易误伤的）：")
    for coverage, gid, query in sorted(answerable)[:5]:
        print(f"  {coverage:.2f}  {gid}  {query}")
    print("\n无答案题的覆盖率（门禁最该拦住的）：")
    for coverage, gid, query in sorted(refusals, reverse=True):
        print(f"  {coverage:.2f}  {gid}  {query}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
