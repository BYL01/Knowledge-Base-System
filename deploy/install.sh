#!/usr/bin/env bash
# 离线安装：把评测跑批装到服务器上。全程不联网。
#
# 用法（在解压出来的目录里）：
#   ./install.sh                  # 装到 /opt/rag-eval
#   INSTALL_DIR=/srv/rag-eval ./install.sh
#   ./install.sh --with-timer     # 顺带装 systemd 每日定时任务
#
# 前置条件：server 上有 Python 3.9+，且当前用户有写 INSTALL_DIR 的权限。
set -euo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR="${INSTALL_DIR:-/opt/rag-eval}"
RUN_USER="${RUN_USER:-$(id -un)}"
LOG_DIR="${LOG_DIR:-/var/log/rag-eval}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WITH_TIMER=0

# root 环境通常没装 sudo，先判断身份，避免脚本卡在 sudo 上
if [[ "$(id -u)" == "0" ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

for arg in "$@"; do
  case "$arg" in
    --with-timer) WITH_TIMER=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "未知参数：$arg（支持 --with-timer）"; exit 2 ;;
  esac
done

echo "== 1/7 检查 Python =="
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "找不到 $PYTHON_BIN，可用 PYTHON_BIN=/usr/bin/python3.11 指定"; exit 1; }
"$PYTHON_BIN" - <<'PYEOF'
import sys
if sys.version_info < (3, 9):
    sys.exit(f"需要 Python 3.9+，当前 {sys.version.split()[0]}")
print("  OK", sys.version.split()[0], sys.executable)
PYEOF

echo "== 2/7 拷贝代码到 $INSTALL_DIR =="
mkdir -p "$INSTALL_DIR"
if [[ "$(readlink -f "$PKG_ROOT")" != "$(readlink -f "$INSTALL_DIR")" ]]; then
  for item in src scripts deploy configs data tests conftest.py requirements.txt README.md .env.example wheels; do
    if [[ -e "$PKG_ROOT/$item" ]]; then
      cp -a "$PKG_ROOT/$item" "$INSTALL_DIR/"
    fi
  done
fi
chmod +x "$INSTALL_DIR"/deploy/*.sh
echo "  OK"

echo "== 3/7 离线安装依赖 =="
VENV="$INSTALL_DIR/venv"
if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV" || {
    echo "  venv 创建失败（可能缺 python3-venv），退回 --user 安装"
    "$PYTHON_BIN" -m pip install --user --no-index --find-links "$PKG_ROOT/wheels" -r "$INSTALL_DIR/requirements.txt"
    VENV=""
  }
fi
if [[ -n "$VENV" ]]; then
  "$VENV/bin/python" -m pip install --no-index --find-links "$PKG_ROOT/wheels" -r "$INSTALL_DIR/requirements.txt" >/dev/null
  echo "  OK 虚拟环境：$VENV"
fi

echo "== 4/7 准备配置文件 =="
if [[ ! -f "$INSTALL_DIR/.env" ]]; then
  cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
  chmod 600 "$INSTALL_DIR/.env"
  echo "  已生成 $INSTALL_DIR/.env，把 DEEPSEEK_API_KEY 填进去（这台机器没有外网就留空）"
else
  chmod 600 "$INSTALL_DIR/.env"
  echo "  已存在 $INSTALL_DIR/.env，保持不变"
fi
mkdir -p "$LOG_DIR" 2>/dev/null || { LOG_DIR="$INSTALL_DIR/logs"; mkdir -p "$LOG_DIR"; echo "  /var/log 不可写，日志改用 $LOG_DIR"; }

echo "== 5/7 构建索引 =="
PY="${VENV:+$VENV/bin/python}"; PY="${PY:-$PYTHON_BIN}"
( cd "$INSTALL_DIR" && "$PY" scripts/build_index.py )

echo "== 6/7 跑冒烟测试 =="
( cd "$INSTALL_DIR" && "$PY" -m pytest tests -q )

echo "== 7/7 定时任务 =="
if [[ "$WITH_TIMER" == "1" ]]; then
  if ! command -v systemctl >/dev/null 2>&1; then
    echo "  这台机器没有 systemd。改用 cron，加这一行："
    echo "  30 2 * * * cd $INSTALL_DIR && INSTALL_DIR=$INSTALL_DIR LOG_DIR=$LOG_DIR ./deploy/run_eval.sh"
  else
    UNIT_DIR="/etc/systemd/system"
    for unit in rag-eval.service rag-eval.timer; do
      sed -e "s|__INSTALL_DIR__|$INSTALL_DIR|g" -e "s|__RUN_USER__|$RUN_USER|g" -e "s|__LOG_DIR__|$LOG_DIR|g" \
        "$INSTALL_DIR/deploy/systemd/$unit" > "/tmp/$unit"
      $SUDO cp "/tmp/$unit" "$UNIT_DIR/$unit"
    done
    $SUDO systemctl daemon-reload
    $SUDO systemctl enable --now rag-eval.timer
    echo "  OK 已启用每日 02:30 跑批（也装好了 service，可单独手动触发）"
  fi
else
  echo "  跳过。需要时重跑：./install.sh --with-timer"
fi

cat <<EOF

安装完成。
  代码目录：$INSTALL_DIR
  运行方式：cd $INSTALL_DIR && ./deploy/run_eval.sh
  手动触发：sudo systemctl start rag-eval.service
  查看结果：sudo journalctl -u rag-eval.service -n 50
            cat $LOG_DIR/eval-*.log
EOF
