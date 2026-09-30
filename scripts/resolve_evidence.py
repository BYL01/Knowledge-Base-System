"""把 golden 集里的 evidence（原文依据）对齐成真实的 gold_chunk_ids。

人工标注时只写"答案出自哪句原文"，chunk id 由脚本按索引算出来，
这样切块参数一变，重跑一次就能重新对齐，不会出现标注和索引对不上的情况。

用法：
    python scripts/resolve_evidence.py                     # 默认处理配置里的 golden 文件
    python scripts/resolve_evidence.py --golden data/golden/golden_v1.jsonl
    python scripts/resolve_evidence.py --check             # 只检查，不写回
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:  # pragma: no cover
    pass

from src.rag.config import load_config  # noqa: E402
from src.rag.store import build_store  # noqa: E402


def normalize(text: str) -> str:
    """去掉所有空白，避免空格、换行差异导致匹配失败。"""
    return re.sub(r"\s+", "", text)


def load_chunks(cfg) -> list[dict]:
    store = build_store(cfg)
    if not store.exists():
        raise SystemExit("索引不存在，先跑 python scripts/build_index.py")
    chunks, _, _ = store.load()
    return [chunk.to_dict() for chunk in chunks]


def resolve_item(item: dict, chunks: list[dict]) -> tuple[list[str], list[str]]:
    evidence = item.get("evidence")
    if not evidence:
        return [], []
    snippets = [evidence] if isinstance(evidence, str) else list(evidence)

    matched: list[str] = []
    missing: list[str] = []
    for snippet in snippets:
        needle = normalize(snippet)
        hits = [c["chunk_id"] for c in chunks if needle in normalize(c["text"])]
        if not hits:
            missing.append(snippet[:40])
        matched.extend(hits)
    return list(dict.fromkeys(matched)), missing


def main() -> int:
    parser = argparse.ArgumentParser(description="按 evidence 对齐 gold_chunk_ids")
    parser.add_argument("--config", default=None)
    parser.add_argument("--golden", default=None)
    parser.add_argument("--check", action="store_true", help="只检查不写回")
    args = parser.parse_args()

    cfg = load_config(args.config)
    golden_path = Path(args.golden) if args.golden else cfg.path_of("golden.path")
    if not golden_path.is_absolute():
        golden_path = Path.cwd() / golden_path

    chunks = load_chunks(cfg)
    rows = [json.loads(line) for line in golden_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    resolved = 0
    problems: list[str] = []
    for item in rows:
        ids, missing = resolve_item(item, chunks)
        if missing:
            problems.append(f"{item['id']} 未匹配到原文：{missing}")
        if ids:
            item["gold_chunk_ids"] = ids
            resolved += 1
        elif not item.get("expect_refusal"):
            problems.append(f"{item['id']} 有答案却没有对齐到任何 chunk")

    print(f"共 {len(rows)} 条，对齐 {resolved} 条，问题 {len(problems)} 条")
    for problem in problems:
        print(f"  ! {problem}")

    if problems:
        print("\n先把上面的问题修掉再写回，否则 gold_chunk_ids 会不准。")
        return 1

    if args.check:
        print("检查通过（未写回文件）")
        return 0

    with golden_path.open("w", encoding="utf-8") as fh:
        for item in rows:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"已写回：{golden_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
