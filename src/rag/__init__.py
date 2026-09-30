"""电商知识库问答平台 —— RAG 评测基线。

分层约定（后面的评测代码依赖这个边界，别打散）：
    ingest   : 语料 -> 切块 -> 向量 -> 落库
    retriever: query -> 命中的 chunk（可直接单测检索指标）
    generator: query + contexts -> 答案 + 引用（可直接单测忠实度/安全指标）
    pipeline : 只做编排，把一次问答的全部中间产物落成 RunRecord
"""

from .config import Config, PROJECT_ROOT, load_config
from .schema import Chunk, RunRecord, ScoredChunk

__all__ = [
    "Config",
    "PROJECT_ROOT",
    "load_config",
    "Chunk",
    "ScoredChunk",
    "RunRecord",
]
