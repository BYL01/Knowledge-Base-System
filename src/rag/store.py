"""向量库适配层。换库只改这里 + configs 里的 store.backend。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .schema import Chunk


def compute_index_version(chunks: list[Chunk]) -> str:
    """索引指纹：内容或切分方式一变，版本就变。后面挂 DVC / 复用缓存靠它。"""
    digest = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda c: c.chunk_id):
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(hashlib.sha256(chunk.text.encode("utf-8")).digest())
    return digest.hexdigest()[:12]


class LocalVectorStore:
    """numpy + jsonl 落盘。零第三方依赖，Windows 直接可用。"""

    name = "local"

    def __init__(self, path: Path):
        self.path = Path(path)

    @property
    def vectors_file(self) -> Path:
        return self.path / "vectors.npy"

    @property
    def chunks_file(self) -> Path:
        return self.path / "chunks.jsonl"

    @property
    def manifest_file(self) -> Path:
        return self.path / "manifest.json"

    def exists(self) -> bool:
        return self.vectors_file.exists() and self.chunks_file.exists()

    def write(self, chunks: list[Chunk], vectors: np.ndarray, manifest: dict) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        np.save(self.vectors_file, vectors.astype(np.float32))
        with self.chunks_file.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        with self.manifest_file.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)

    def load(self) -> tuple[list[Chunk], np.ndarray, dict]:
        if not self.exists():
            raise FileNotFoundError(
                f"索引不存在：{self.path}。先跑 python scripts/build_index.py"
            )
        vectors = np.load(self.vectors_file)
        chunks: list[Chunk] = []
        with self.chunks_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    chunks.append(Chunk(**json.loads(line)))
        manifest = {}
        if self.manifest_file.exists():
            with self.manifest_file.open("r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        return chunks, vectors, manifest

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        chunks, vectors, _ = self.load()
        if not chunks:
            return []
        scores = vectors @ query_vector.reshape(-1)
        order = np.argsort(-scores)[:top_k]
        return [(chunks[i], float(scores[i])) for i in order]


class ChromaVectorStore:
    """可选后端。装 chromadb 并把 store.backend 改成 chroma 即可。

    Milvus Lite 不支持 Windows，本地先用 Chroma 顶替；
    上服务器/CI 时在这里再补一个 MilvusVectorStore，接口保持一致。
    """

    name = "chroma"

    def __init__(self, path: Path, collection: str = "ecom_kb"):
        try:
            import chromadb  # type: ignore
        except ImportError as exc:  # pragma: no cover - 取决于本地是否安装
            raise RuntimeError(
                "store.backend=chroma 需要先 pip install chromadb"
            ) from exc
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def exists(self) -> bool:
        return self._collection.count() > 0

    def write(self, chunks: list[Chunk], vectors: np.ndarray, manifest: dict) -> None:
        if chunks:
            self._collection.upsert(
                ids=[c.chunk_id for c in chunks],
                embeddings=vectors.tolist(),
                documents=[c.text for c in chunks],
                metadatas=[{"doc_id": c.doc_id, "source": c.source, "position": c.position} for c in chunks],
            )

    def load(self) -> tuple[list[Chunk], np.ndarray, dict]:
        raise NotImplementedError("Chroma 后端不提供全量载入，请直接用 search()")

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        result = self._collection.query(query_embeddings=[query_vector.tolist()], n_results=top_k)
        hits: list[tuple[Chunk, float]] = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=meta.get("doc_id", ""),
                source=meta.get("source", ""),
                text=text,
                position=int(meta.get("position", 0)),
            )
            hits.append((chunk, 1.0 - float(distance)))
        return hits


def build_store(cfg):
    backend = str(cfg.get("store.backend", "local")).lower()
    path = cfg.path_of("store.path")
    if backend == "local":
        return LocalVectorStore(path)
    if backend == "chroma":
        return ChromaVectorStore(path, collection=str(cfg.get("store.collection", "ecom_kb")))
    raise ValueError(f"未支持的 store.backend：{backend}")
