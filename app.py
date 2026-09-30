"""极简交互 Demo：输入问题 -> 答案 / 引用 / 拒答判定。

本地运行：
    pip install -r requirements-demo.txt
    streamlit run app.py

默认零 API key 也能跑（hashing 检索 + extractive 生成）。
配置了 DEEPSEEK_API_KEY 后可以切到语义判官，体验"知识库里没有就拒答"。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.config import load_config, load_env_file  # noqa: E402
from src.rag.ingest import build_index  # noqa: E402
from src.rag.llm import ChatClient  # noqa: E402
from src.rag.pipeline import REFUSAL_ANSWER, RAGPipeline  # noqa: E402

st.set_page_config(page_title="知识库问答 Demo", page_icon="📚", layout="wide")

SAMPLES = [
    "未支付的订单可以直接取消吗？",
    "生鲜商品支持七天无理由退货吗？",
    "青海属于偏远地区吗？",
    "可以用花呗分期支付吗？",
]


def bridge_secrets() -> None:
    """把 Streamlit Cloud 的 secrets 注入环境变量，判官就能直接读。"""
    try:
        secrets = st.secrets
    except Exception:
        return
    for key in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        try:
            if key in secrets and not os.environ.get(key):
                os.environ[key] = str(secrets[key])
        except Exception:
            continue


# 先读本地 .env（已存在的环境变量优先），再叠加 Streamlit Cloud 的 secrets
load_env_file()
bridge_secrets()


@st.cache_resource(show_spinner="首次运行：构建索引…")
def load_manifest() -> dict:
    """索引不入库，首次运行就地重建（约 0.5 秒）。"""
    cfg = load_config()
    index_dir = cfg.path_of("store.path")
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        build_index(cfg, verbose=False)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="加载链路…")
def get_pipeline(mode: str, min_coverage: float, top_k: int) -> RAGPipeline:
    cfg = load_config()
    cfg.set("answerability.mode", mode)
    cfg.set("answerability.min_query_coverage", float(min_coverage))
    cfg.set("retriever.top_k", int(top_k))
    return RAGPipeline.from_config(cfg)


def judge_available() -> bool:
    return ChatClient(provider="deepseek").available


manifest = load_manifest()
has_key = judge_available()

st.title("📚 知识库问答 · RAG 评测 Demo")
st.caption("检索 → 生成 → 拒答判定，三层各自独立，中间产物全部可见。默认后端零 API key、输出确定。")

with st.sidebar:
    st.header("参数")
    if has_key:
        mode = st.radio(
            "可答性门禁",
            ["llm", "heuristic", "off"],
            index=0,
            help="llm：语义判官（需 DeepSeek key），知识库里没有就拒答 ｜ heuristic：关键词覆盖率 ｜ off：不判定",
        )
    else:
        mode = st.radio(
            "可答性门禁",
            ["heuristic", "off"],
            index=0,
            help="未配置 DEEPSEEK_API_KEY，语义判官不可用。heuristic：关键词覆盖率 ｜ off：不判定",
        )
        st.info("未检测到 `DEEPSEEK_API_KEY`，已回退到零成本模式。想要「知识库里没有就拒答」请配置该 key。")

    min_coverage = st.slider("覆盖率阈值（仅 heuristic）", 0.0, 1.0, 0.30, 0.05)
    top_k = st.slider("检索条数 top_k", 1, 8, 4)

    st.divider()
    st.caption(f"索引版本 `{manifest.get('index_version')}`")
    st.caption(f"语料 {manifest.get('doc_count')} 篇 · chunk {manifest.get('chunk_count')} 个")

if "question" not in st.session_state:
    st.session_state.question = SAMPLES[0]

run_now = False
st.markdown("**不知道问什么？点一个：**")
sample_cols = st.columns(len(SAMPLES))
for col, sample in zip(sample_cols, SAMPLES):
    if col.button(sample, width="stretch"):
        st.session_state.question = sample
        run_now = True

with st.form("ask", clear_on_submit=False):
    st.text_input("你的问题", key="question")
    submitted = st.form_submit_button("提问", type="primary")

if submitted or run_now:
    if not st.session_state.question.strip():
        st.warning("先输入一个问题。")
    else:
        pipeline = get_pipeline(mode, min_coverage, top_k)
        with st.spinner("检索 + 生成中…"):
            st.session_state.result = pipeline.answer(st.session_state.question.strip()).to_dict()

record = st.session_state.get("result")

if record:
    decision = record.get("extra", {}).get("answerability") or {}
    refused = record["answer"] == REFUSAL_ANSWER

    st.divider()
    st.markdown(f"**问：** {record['query']}")

    if refused:
        st.warning(f"🚫 拒答：{record['answer']}", icon="🚫")
    else:
        st.success(record["answer"], icon="✅")

    meta_cols = st.columns(4)
    meta_cols[0].metric("耗时", f"{record['latency_ms']} ms")
    meta_cols[1].metric("检索命中", f"{len(record['retrieved'])} 条")
    meta_cols[2].metric("引用", f"{len(record['citations'])} 条")
    meta_cols[3].metric("悬空引用", len(record["extra"].get("dangling_citations", [])))

    if decision:
        if decision.get("error"):
            st.error(f"判官出错，按 `on_error` 兜底：{decision['error']}")
        else:
            st.caption(f"判定依据（{decision.get('mode')} / {decision.get('source')}）：{decision.get('reason', '')}")

    if record["citations"]:
        st.markdown("**引用来源：** " + " ".join(f"`{cid}`" for cid in record["citations"]))

    with st.expander("看检索层返回了什么", expanded=False):
        st.dataframe(
            [
                {
                    "chunk_id": item["chunk_id"],
                    "score": round(item["score"], 4),
                    "来源": item["source"],
                }
                for item in record["retrieved"]
            ],
            width="stretch",
            hide_index=True,
        )

    with st.expander("看引用对应的原文", expanded=False):
        cited = set(record["citations"])
        for item in record["retrieved"]:
            if item["chunk_id"] in cited:
                st.markdown(f"**{item['chunk_id']}**（{item['source']}）")
                st.code(item["text"], language=None)

    with st.expander("看发给生成层的 prompt", expanded=False):
        st.code(record["prompt"] or "（拒答时不会构造 prompt）", language=None)
