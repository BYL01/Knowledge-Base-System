#!/usr/bin/env bash
# 环境自检：装完先跑这个，确认服务器上缺什么、DeepSeek 通不通。
set -uo pipefail

INSTALL_DIR="${INSTALL_DIR:-/opt/rag-eval}"
cd "$INSTALL_DIR"

PY="$INSTALL_DIR/venv/bin/python"
[[ -x "$PY" ]] || PY="${PYTHON_BIN:-python3}"

echo "--- Python ---"
"$PY" -c 'import sys; print(sys.executable); print(sys.version)'

echo
echo "--- 依赖 ---"
"$PY" - <<'EOF'
for name in ("numpy", "yaml", "requests", "pytest"):
    try:
        module = __import__(name)
        print(f"  OK   {name} {getattr(module, '__version__', '')}")
    except ImportError as exc:
        print(f"  FAIL {name}: {exc}")
EOF

echo
echo "--- 索引 ---"
if [[ -f data/index/baseline/manifest.json ]]; then
  "$PY" -c 'import json;m=json.load(open("data/index/baseline/manifest.json",encoding="utf-8"));print("  OK   chunk",m["chunk_count"],"| index_version",m["index_version"])'
else
  echo "  未构建，稍后跑 python scripts/build_index.py"
fi

echo
echo "--- DeepSeek 连通性 ---"
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  if [[ -f .env ]]; then
    set -a; source .env; set +a
  fi
fi
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "  未配置 DEEPSEEK_API_KEY（.env 里填），判官将走缓存或兜底策略"
else
  "$PY" - <<'EOF'
import os
import requests
try:
    resp = requests.post(
        os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1") + "/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"},
        json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1},
        timeout=20,
    )
    print("  OK   可访问 DeepSeek" if resp.ok else f"  FAIL HTTP {resp.status_code}: {resp.text[:200]}")
except Exception as exc:
    print(f"  FAIL 无法访问 DeepSeek：{exc}")
    print("       如果是无外网机器：把 judge.cache_only 设为 true，判官只读本地缓存")
EOF
fi

echo
echo "--- 磁盘 ---"
df -h "$INSTALL_DIR" | tail -1
