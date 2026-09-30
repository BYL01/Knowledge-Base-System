"""语料 -> 切块 -> 向量 -> 落库。"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from pathlib import Path

import numpy as np

from .embedder import build_embedder
from .schema import Chunk
from .store import build_store, compute_index_version

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_SENTENCE_END = "。！？!?；;\n"


def load_documents(corpus_dir: Path, pattern: str = "*.md") -> list[dict]:
    if not corpus_dir.exists():
        raise FileNotFoundError(f"语料目录不存在：{corpus_dir}")
    documents = []
    for path in sorted(corpus_dir.glob(pattern)):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append({"doc_id": path.stem, "source": path.name, "text": text})
    if not documents:
        raise ValueError(f"{corpus_dir} 下没有匹配 {pattern} 的语料")
    return documents


def _blocks_with_headings(text: str) -> list[tuple[str, str]]:
    """按空行切段，并记录每段所属的最近一级标题。"""
    blocks: list[tuple[str, str]] = []
    heading = ""
    for raw in re.split(r"\n\s*\n", text):
        block = raw.strip()
        if not block:
            continue
        lines = block.splitlines()
        if _HEADING.match(lines[0]):
            heading = _HEADING.match(lines[0]).group(2).strip()
            lines = lines[1:]
            block = "\n".join(lines).strip()
            if not block:
                continue
        blocks.append((heading, block))
    return blocks


def _render_block(items: list[tuple[str, str]]) -> str:
    heading = items[0][0]
    body = "\n".join(body for _, body in items)
    return f"{heading}\n{body}" if heading else body


def chunk_paragraphs(blocks: list[tuple[str, str]], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[tuple[str, str]] = []
    current_len = 0
    for heading, body in blocks:
        if current and current_len + len(body) > size:
            chunks.append(_render_block(current))
            carried: list[tuple[str, str]] = []
            carried_len = 0
            for item in reversed(current):
                if carried_len + len(item[1]) <= overlap:
                    carried.insert(0, item)
                    carried_len += len(item[1])
                else:
                    break
            current, current_len = carried, carried_len
        current.append((heading, body))
        current_len += len(body)
    if current:
        chunks.append(_render_block(current))
    return [c for c in chunks if c.strip()]


def chunk_fixed(text: str, size: int, overlap: int) -> list[str]:
    """定长滑窗，尽量在句末切开，避免把一句话劈成两半。"""
    step = max(1, size - overlap)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            window = text[start:end]
            cut = max((window.rfind(mark) for mark in _SENTENCE_END), default=-1)
            if cut > size * 0.5:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + step)
    return chunks


def clean_markdown(text: str) -> str:
    return _HEADING.sub(lambda m: m.group(0), text)


def build_chunks(documents: list[dict], size: int, overlap: int, strategy: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        if strategy == "fixed":
            pieces = chunk_fixed(document["text"], size, overlap)
        elif strategy == "paragraph":
            pieces = chunk_paragraphs(_blocks_with_headings(document["text"]), size, overlap)
        else:
            raise ValueError(f"未支持的 chunking.strategy：{strategy}")
        for index, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{document['doc_id']}#{index:02d}",
                    doc_id=document["doc_id"],
                    source=document["source"],
                    text=piece,
                    position=index,
                )
            )
    return chunks


def build_index(cfg, *, verbose: bool = True) -> dict:
    corpus_dir = cfg.path_of("corpus.dir")
    pattern = str(cfg.get("corpus.glob", "*.md"))
    size = int(cfg.get("chunking.chunk_size", 320))
    overlap = int(cfg.get("chunking.chunk_overlap", 80))
    strategy = str(cfg.get("chunking.strategy", "paragraph"))

    documents = load_documents(corpus_dir, pattern)
    chunks = build_chunks(documents, size, overlap, strategy)

    embedder = build_embedder(cfg)
    vectors = embedder.encode([c.text for c in chunks]) if chunks else np.zeros((0, 0), np.float32)
    if chunks and vectors.shape[0] != len(chunks):
        raise RuntimeError("向量条数与 chunk 数不一致，检查 embedder 实现")

    store = build_store(cfg)
    index_version = compute_index_version(chunks)
    manifest = {
        "project": cfg.get("project"),
        "baseline_version": cfg.get("baseline_version"),
        "index_version": index_version,
        "built_at": dt.datetime.now().isoformat(timespec="seconds"),
        "chunk_count": len(chunks),
        "doc_count": len(documents),
        "chunking": {"strategy": strategy, "chunk_size": size, "chunk_overlap": overlap},
        "embedding": embedder.signature(),
        "store_backend": store.name,
        "corpus": [
            {
                "doc_id": d["doc_id"],
                "source": d["source"],
                "sha1": hashlib.sha1(d["text"].encode("utf-8")).hexdigest()[:12],
            }
            for d in documents
        ],
    }
    store.write(chunks, vectors, manifest)

    if verbose:
        print(f"[build_index] 语料 {len(documents)} 篇 -> chunk {len(chunks)} 个")
        print(f"[build_index] 向量后端 {embedder.signature()} | 存储 {store.name}")
        print(f"[build_index] index_version = {index_version}")
    return manifest
