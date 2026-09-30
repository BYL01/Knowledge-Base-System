"""构建索引：python scripts/build_index.py [--config configs/baseline.yaml]

索引可随时重建，产物在 data/index/<name>/ 下，已被 .gitignore 忽略。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:  # Windows 控制台默认 cp936，中文输出会乱码
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:  # pragma: no cover
    pass

from src.rag.config import load_config, load_env_file  # noqa: E402
from src.rag.ingest import build_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 RAG 基线索引")
    parser.add_argument("--config", default=None, help="配置文件路径，默认 configs/baseline.yaml")
    args = parser.parse_args()

    load_env_file()
    cfg = load_config(args.config)
    build_index(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
