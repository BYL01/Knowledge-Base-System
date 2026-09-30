"""打离线部署包。在有外网的机器上运行，产物拷到无外网服务器上安装。

用法：
    py -3.11 deploy/build_bundle.py                      # Linux x86_64 + CPython 3.11
    py -3.11 deploy/build_bundle.py --python-version 310

产物：dist/rag-eval-offline-<日期>.tar.gz

用 Python 而不是 tar 命令打包，有两个实打实的理由：
1. 路径里有空格时，PowerShell 传给原生 tar.exe 的参数会被拆坏（实测踩过）；
2. Windows 上没有执行位，tar 打出来的 .sh 在 Linux 解压后不可执行。
   这里显式把权限写进压缩包，服务器上解开就能直接跑。
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
STAGE = DIST / "stage" / "rag-eval-offline"
APP = STAGE / "app"
WHEELS = APP / "wheels"

PACKAGES = ["numpy", "PyYAML", "requests", "pytest"]
COPY_ITEMS = [
    "src",
    "scripts",
    "deploy",
    "configs",
    "data",
    "tests",
    "conftest.py",
    "requirements.txt",
    "README.md",
    ".env.example",
    ".gitignore",
]
# index 由服务器上自己重建，runs 是本地产物，都不进包。
# judge_cache 默认也不进包：让服务器第一次跑就是真实调用，顺便验证它自己的连通性。
# 确实需要离线复用时加 --with-judge-cache，把本地预热好的判定结果打进去。
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git", "index", "runs", "judge_cache"}

WRAPPER = """#!/usr/bin/env bash
# Offline bundle entry point.
set -euo pipefail
cd "$(dirname "$0")/app"
exec ./deploy/install.sh "$@"
"""


def download_wheels(python_bin: str, platform: str, version: str) -> None:
    command = [
        python_bin, "-m", "pip", "download", "--quiet",
        "--only-binary=:all:", "--dest", str(WHEELS),
        "--platform", platform, "--platform", "any",
        "--python-version", version, "--implementation", "cp",
        "--abi", f"cp{version}", "--abi", "none",
        *PACKAGES,
    ]
    subprocess.run(command, check=True)
    wheels = list(WHEELS.glob("*.whl"))
    size_mb = sum(w.stat().st_size for w in wheels) / 1024 / 1024
    print(f"   下载 {len(wheels)} 个 wheel，{size_mb:.1f} MB")


def ignore(directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in SKIP_DIRS}


def copy_app() -> None:
    for item in COPY_ITEMS:
        source = ROOT / item
        if not source.exists():
            continue
        target = APP / item
        if source.is_dir():
            shutil.copytree(source, target, ignore=ignore, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)


def make_tar(bundle: Path) -> None:
    def fix_modes(info: tarfile.TarInfo) -> tarfile.TarInfo:
        if info.isdir() or info.name.endswith(".sh"):
            info.mode = 0o755
        else:
            info.mode = 0o644
        info.uid = info.gid = 0
        info.uname = info.gname = "root"
        return info

    with tarfile.open(bundle, "w:gz") as archive:
        archive.add(STAGE, arcname="rag-eval-offline", filter=fix_modes)


def main() -> int:
    parser = argparse.ArgumentParser(description="打离线部署包")
    parser.add_argument("--python-version", default="311", help="目标 CPython 版本，如 311 / 310")
    parser.add_argument("--platform", default="manylinux2014_x86_64", help="目标平台 tag")
    parser.add_argument("--python-bin", default=sys.executable, help="本机用于下载 wheel 的 Python")
    parser.add_argument("--with-judge-cache", action="store_true", help="把本地预热的判官缓存一起打进包")
    args = parser.parse_args()

    if args.with_judge_cache:
        SKIP_DIRS.discard("judge_cache")

    print("== 1/4 清理旧产物 ==")
    if STAGE.exists():
        shutil.rmtree(STAGE)
    WHEELS.mkdir(parents=True)

    print(f"== 2/4 下载依赖（{args.platform} / py{args.python_version}）==")
    download_wheels(args.python_bin, args.platform, args.python_version)

    print("== 3/4 打包代码 ==")
    copy_app()
    (STAGE / "install.sh").write_text(WRAPPER, encoding="utf-8", newline="\n")
    print(f"   代码 {sum(1 for _ in APP.rglob('*') if _.is_file())} 个文件")

    print("== 4/4 生成 tar.gz ==")
    bundle = DIST / f"rag-eval-offline-{dt.date.today():%Y%m%d}.tar.gz"
    if bundle.exists():
        bundle.unlink()
    make_tar(bundle)
    print(f"\n打包完成：{bundle}")
    print(f"大小：{bundle.stat().st_size / 1024 / 1024:.1f} MB")
    print("服务器上：tar xzf <包名> && cd rag-eval-offline && ./install.sh --with-timer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
