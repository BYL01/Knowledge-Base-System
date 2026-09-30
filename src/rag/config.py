"""配置加载：把 configs/*.yaml 变成可访问的对象，并把相对路径解析到项目根目录。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "baseline.yaml"


class Config:
    """支持点号取值的配置对象，例如 cfg.get("retriever.top_k", 4)。"""

    def __init__(self, raw: dict[str, Any], path: Path):
        self.raw = raw
        self.path = path

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def set(self, dotted: str, value: Any) -> None:
        keys = dotted.split(".")
        node = self.raw
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value

    def path_of(self, dotted: str, default: str | None = None) -> Path:
        """取路径配置项，相对路径统一按项目根目录解析。"""
        value = self.get(dotted, default)
        if value is None:
            raise KeyError(f"配置项不存在：{dotted}")
        candidate = Path(str(value))
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate

    def env(self, name: str, default: str | None = None) -> str | None:
        return os.environ.get(name, default)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"Config(path={self.path}, project={self.get('project')})"


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config(raw, cfg_path)


def load_env_file(path: str | Path | None = None) -> None:
    """极简 .env 读取（不引入 python-dotenv 依赖），已存在的环境变量优先。"""
    env_path = Path(path) if path else PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    with env_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def db_settings() -> dict[str, Any]:
    """测试环境数据库连接参数：统一从 .env / 环境变量取，代码里不写死连接串。

    没配 DB_HOST 时返回空字符串而不是抛错，这样不连库的环境（CI、离线包）
    照样能跑评测链路；调用方自己决定是报错还是跳过。
    """
    load_env_file()
    port_text = os.environ.get("DB_PORT", "3306")
    if not port_text.isdigit():
        raise ValueError(f"DB_PORT 必须是整数，当前值：{port_text!r}")
    return {
        "host": os.environ.get("DB_HOST", ""),
        "port": int(port_text),
        "user": os.environ.get("DB_USER", ""),
        "password": os.environ.get("DB_PASSWORD", ""),
        "database": os.environ.get("DB_NAME", ""),
        "charset": os.environ.get("DB_CHARSET", "utf8mb4"),
    }
