"""Embedding 后端。

默认 hashing：纯本地、确定性、不需要 API key。
它不是"好"的语义向量，但足够让检索链路先跑通，并且让 CI 里的评测结果可复现。
换成真实模型只需改 configs 里的 embedding.backend，评测代码不用动。
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np

_LATIN = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """中英混排的粗粒度分词：英文按词，中文按字 + 相邻字二元组。"""
    lowered = text.lower()
    tokens = _LATIN.findall(lowered)
    cjk = _CJK.findall(lowered)
    tokens.extend(cjk)
    tokens.extend(a + b for a, b in zip(cjk, cjk[1:]))
    return tokens or ["<empty>"]


class HashingEmbedder:
    """带符号的特征哈希向量，L2 归一化后可用点积当余弦相似度。"""

    name = "hashing"

    def __init__(self, dim: int = 512):
        self.dim = int(dim)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._encode_one(t) for t in texts])

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token, count in Counter(tokenize(text)).items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign * (1.0 + math.log(count))
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def signature(self) -> dict[str, object]:
        return {"backend": self.name, "dim": self.dim}


class OpenAIEmbedder:
    """真实语义向量。需要 OPENAI_API_KEY，走 HTTP 直连以免多装一个 SDK。"""

    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small", base_url: str | None = None):
        import requests  # 局部导入：没用到这个后端时不产生依赖

        self._requests = requests
        self.model = model
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = __import__("os").environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("embedding.backend=openai 需要先设置 OPENAI_API_KEY")

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        response = self._requests.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        vectors = [item["embedding"] for item in payload["data"]]
        array = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return array / np.clip(norms, 1e-12, None)

    def signature(self) -> dict[str, object]:
        return {"backend": self.name, "model": self.model}


def build_embedder(cfg) -> HashingEmbedder | OpenAIEmbedder:
    backend = str(cfg.get("embedding.backend", "hashing")).lower()
    if backend == "hashing":
        return HashingEmbedder(dim=int(cfg.get("embedding.dim", 512)))
    if backend == "openai":
        return OpenAIEmbedder(
            model=str(cfg.get("embedding.model", "text-embedding-3-small")),
            base_url=cfg.env("OPENAI_BASE_URL"),
        )
    raise ValueError(f"未支持的 embedding.backend：{backend}")
