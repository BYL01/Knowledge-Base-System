#!/usr/bin/env bash
# 单次评测跑批：建索引 -> 跑 golden 集 -> 质量门禁。
# 退出码即门禁结果：0 通过，非 0 不通过（CI / systemd 可直接用）。
set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-/opt/rag-eval}"
LOG_DIR="${LOG_DIR:-/var/log/rag-eval}"
REBUILD_INDEX="${REBUILD_INDEX:-1}"

cd "$INSTALL_DIR"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval-$(date +%Y%m%d).log"

# 环境变量：优先用 .env 里的（systemd 也会注入一份）
if [[ -f "$INSTALL_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$INSTALL_DIR/.env"
  set +a
fi

if [[ -x "$INSTALL_DIR/venv/bin/python" ]]; then
  PY="$INSTALL_DIR/venv/bin/python"
else
  PY="${PYTHON_BIN:-python3}"
fi

set +e
{
  echo "===== $(date '+%F %T') 评测跑批开始 ====="
  if [[ "$REBUILD_INDEX" == "1" ]]; then
    "$PY" scripts/build_index.py
  fi
  "$PY" scripts/run_batch.py
  "$PY" scripts/gate.py
  echo "===== $(date '+%F %T') 通过 ====="
} 2>&1 | tee -a "$LOG_FILE"
# tee 会吃掉退出码，用 PIPESTATUS 取真实结果
STATUS=${PIPESTATUS[0]}
set -e
exit "$STATUS"
