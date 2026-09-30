# 电商知识库问答平台 · RAG 评测基线

这是一个 **RAG 自动化评测框架的地基**：先把"被评测对象"跑通，让链路可以被拆开单独测量，
再用一套标注基准把它的真实能力量化出来。评测框架的价值不在代码量，而在于每个指标都能拿到它需要的中间产物。

当前规模：**24 篇语料 / 49 个 chunk / 74 条人工标注基准**。

## 快速开始

本机没有安装 `python` 命令（Windows Store 占位符），统一用 `py -3.11`：

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 scripts/build_index.py     # 灌库：语料 -> chunk -> 向量
py -3.11 scripts/resolve_evidence.py  # 按原文依据对齐 gold_chunk_ids（改语料/切块后重跑）
py -3.11 scripts/run_batch.py       # 跑 golden 集，落盘中间产物 + 打印指标
py -3.11 scripts/gate.py            # 质量门禁：指标不达标返回非 0，CI 直接当卡口
py -3.11 -m pytest tests -q         # 冒烟测试（可直接当 CI 第一道门禁）
```

跑批产物：

- `outputs/runs/<run_id>.jsonl`：每条问答的完整中间产物（query / retrieved / prompt / answer / citations / latency）
- `outputs/runs/<run_id>.summary.json`：本次跑批的指标汇总

## 配置与密钥

所有敏感项（DeepSeek API Key、测试库连接、SSH 私钥路径）**只放在本地 `.env`，不进 git**：

```powershell
Copy-Item .env.example .env   # 然后按需填值
```

- `.env` 已在 `.gitignore` 里，凭据不会随代码提交；服务器上建议 `chmod 600 .env`。
- 默认的 `hashing` embedding + `extractive` 生成**不需要任何 key**，克隆下来直接能跑通链路。
- 文档、脚本、配置里不写死服务器 IP、账号或密钥，需要时一律走环境变量。

## 目录结构

```
configs/baseline.yaml        所有可调参数 + 质量门禁阈值，改一项等于产生一个新的评测版本
data/corpus/                 语料（24 篇电商知识库文档）
data/golden/golden_v0.jsonl  初版 10 条基准（保留作历史对比）
data/golden/golden_v1.jsonl  现行 74 条基准（68 条可答 + 6 条无答案）
src/rag/
    config.py                配置加载
    schema.py                数据契约：Chunk / ScoredChunk / RunRecord
    embedder.py              向量化后端（hashing 默认，openai 可选）
    store.py                 向量库适配（local 默认，chroma 可选）
    ingest.py                切块 + 建索引
    relevance.py             关键词覆盖率（启发式信号）
    llm.py                   OpenAI 兼容的 Chat 客户端（DeepSeek / OpenAI）
    judge.py                 LLM 判官：可答性判定 + 原子 claim 拆解（带缓存与熔断）
    answerability.py         拒答门禁：off / heuristic / llm 三种模式
    retriever.py             ★ 检索层，可单独调用
    generator.py             ★ 生成层，可单独调用
    pipeline.py              编排，产出 RunRecord
scripts/build_index.py       建索引
scripts/resolve_evidence.py  标注辅助：evidence 原文 -> gold_chunk_ids
scripts/run_batch.py         跑批 + 指标
scripts/sweep_refusal.py     拒答阈值扫描（用数据定阈值，而不是拍脑袋）
scripts/gate.py              质量门禁
deploy/                      离线部署包：install.sh / run_eval.sh / systemd / 打包脚本
tests/                       冒烟测试 + 门禁行为测试 + LLM 接入层测试（用本地假服务端）
```

## 三条分层边界（后面写评测代码的前提）

1. `retriever.retrieve(query)` 只返回 `ScoredChunk`，不碰生成。检索指标（doc_hit / chunk_hit / recall@k / MRR）直接挂在这里。
2. `generator.generate(query, contexts)` 只吃上下文出答案，可以喂假上下文单独测。忠实度、安全性、拒答能力都挂在这里。
3. `pipeline.answer()` 不做任何业务逻辑，只负责把一次问答的全部中间产物封成 `RunRecord`。评测、badcase 回流、CI 门禁都消费这份数据，不重跑链路。

`RunRecord.extra.dangling_citations` 会记录"引用了不存在的资料编号"的情况——引用回溯不了就是幻觉引用，这是最廉价的一道忠实度信号。

## 标注工作流

人工标注只写两样东西：**标准答案** 和 **答案出自哪句原文**（`evidence` 字段）。
chunk id 由 `scripts/resolve_evidence.py` 按索引算出来。好处是切块参数一改，
重跑一次就能重新对齐，不会出现"标注和索引对不上"的隐性错误。

```
标注 evidence（原文依据） -> resolve_evidence.py 自动对齐 gold_chunk_ids -> run_batch 度量
```

## 当前表现（v1 数据，DeepSeek 判官已启用）

```
doc_hit_rate        : 0.9706    # 68 条可答题里，2 条没检索到正确文档
chunk_hit_rate      : 0.9559
chunk_recall_at_k   : 0.9412
refusal_accuracy    : 1.0       # 6 条无答案题全部正确拒答
false_refusal_rate  : 0.0441    # 3/68 被误拒，根因见下
dangling_citations  : 0
avg_latency_ms      : 579.9     # 判官调用带来的开销
p95_latency_ms      : 1187
```

## 结论一：拒答要靠语义判官，关键词门禁被数据否决

6 条无答案题全被硬答，第一反应是加相关性阈值。先去量一下数据：

```
可答题最高分的分布：min 0.103  中位 0.321  max 0.697
无答案题最高分：      0.155 ~ 0.276
```

两个区间**完全重叠**，任何绝对分数阈值都会同时误杀和漏放，这条路直接走不通。

退而求其次做"查询关键词覆盖率"门禁（`src/rag/relevance.py`），用 `scripts/sweep_refusal.py` 扫阈值：

| 覆盖率阈值 | 拦住的无答案题 | 误拒的可答题 | 综合准确率 |
|---|---|---|---|
| 0.00（关闭） | 0/6 | 0 | 0.9189 |
| 0.30 | 4/6 | 6 | 0.8919 |
| 0.45 | 6/6 | 19 | 0.7432 |

**每个能拦住拒答题的阈值，误拒的题都更多，净收益为负。** 原因是词汇级信号分不清两件事：
"问法和原文用词不一样"（该答）和"知识库里根本没有这块知识"（该拒）。

两条路都不通之后接 DeepSeek 判官（`answerability.mode: llm`），一次到位：

| 指标 | 门禁关闭（基线） | LLM 判官 |
|---|---|---|
| refusal_accuracy | 0.0 | **1.0** |
| false_refusal_rate | 0.0 | 0.0441 |
| avg_latency_ms | 0.1 | 579.9 |

关键是搞清楚那 3 条误拒到底怪谁。逐条查下来 **3/3 全部是检索层的锅，判官零误判**：

| 用例 | 检索返回 | 答案实际所在 | 判官结论 |
|---|---|---|---|
| g0112 新疆运费 | 订单修改 / 支付 / 物流 | 配送合作方与运费#00 | 正确拒答，资料里确实没有运费金额 |
| g0127 寄修运费到付 | 售后 / 配送 / 发票 | 上门服务与寄修#00 | 正确拒答 |
| g0165 买多了会砍单吗 | 商品限购与库存**#01** | 商品限购与库存**#00** | 正确拒答，#01 里没有限购规则 |

这个结果说明链路是健康的：检索不到证据时判官会拒答，而不是像基线那样硬编一个答案。
**下一步该优化的是检索（换真实 embedding、调切分），不是判官。**

## 结论二：两个真实检索 badcase

```
g0112 新疆的订单需要支付多少运费？   期望 配送合作方与运费.md，实际检索跑到"偏远地区"那篇
g0127 寄修的运费可以到付吗？         "运费"关键词把检索拉到了运费政策文档，漏掉了寄修文档
```

两条都属于"同主题多文档互相干扰"，是扩语料之后才暴露出来的问题。

## 质量门禁

`scripts/gate.py` 读取最近一次跑批汇总，对照 `configs/baseline.yaml` 里的 `gate` 阈值：

```
doc_hit_rate           0.9706 >= 0.90    PASS
chunk_hit_rate         0.9559 >= 0.85    PASS
refusal_accuracy          1.0 >= 0.90    PASS
false_refusal_rate     0.0441 <= 0.05    PASS
dangling_citations          0 <= 0       PASS
p95_latency_ms           1187 <= 2000    PASS
```

门禁现在**通过**（退出码 0）。它之前红过一段时间，卡在 `refusal_accuracy`——那是故意的：
门禁表达的是产品要求而不是当前水平，把阈值调松就等于把问题藏起来。
接入语义判官之后它自己变绿，说明这条红线是有效的。

## 为什么要做成零依赖可跑

默认后端是 `hashing` embedding + `extractive` 生成，**不需要任何 API key，输出完全确定**。
这样做的目的：CI 里结果可复现，换模型时能和基线对比出真实差异；如果一开始就用大模型，指标波动来自模型本身，分不清是链路问题还是模型问题。

换成真实组件只改 `configs/baseline.yaml`，评测层代码一行不动：

```yaml
embedding:
  backend: openai          # 需要 OPENAI_API_KEY
generator:
  backend: openai          # 需要 OPENAI_API_KEY
store:
  backend: chroma          # 需要 pip install chromadb
```

注意：Milvus Lite 不支持 Windows，本机先用 Chroma 或 local 顶替，上服务器/CI 再补一个 `MilvusVectorStore`（接口已预留，见 `store.py`）。

## 测试环境数据库

测试环境的库连接参数统一放在 `.env`（不入库），代码侧用 `src/rag/config.py` 的 `db_settings()` 取，
返回 host / port / user / password / database / charset，省得每个脚本各写一遍连接串：

```python
from src.rag.config import db_settings

conn = pymysql.connect(**db_settings())   # 需要先 pip install pymysql
```

`DB_HOST` 没配时返回空字符串而不是抛错，CI、离线包这类不连库的环境照样能跑评测链路。
连库只用于测试数据准备/清理和写操作的落库校验，**禁止指向线上库**。

## 部署形态

这一阶段的"部署"不是把服务挂上去，而是**把评测跑批变成服务器上的定时任务 + 门禁**：

```
systemd timer  ->  run_batch.py  ->  gate.py  ->  退出码即门禁结果
```

目标环境是**无 Docker、无 Jenkins 的 Linux 服务器**（原本以为没外网，实测能通 pypi 和 DeepSeek），整套走离线安装：

```powershell
py -3.11 deploy/build_bundle.py      # 在本机打离线包（含 Linux 版 wheel）
```

```bash
# 服务器上
tar xzf rag-eval-offline-*.tar.gz && cd rag-eval-offline && ./install.sh --with-timer
```

完整步骤（含无外网时判官怎么做）见 [deploy/README.md](deploy/README.md)。

### 实际部署记录（2026-09-30）

| 项 | 值 |
|---|---|
| 服务器 | 一台 Linux 云主机（Alibaba Cloud Linux 3，x86_64），公网 IP 不在文档里记录 |
| 登录 | 普通运维账号 + RSA 私钥（凭据只放 `.env`，不入库） |
| 运行时 | 系统自带 python3.6.8 不满足要求，`yum install python3.11` 装了 3.11.13 |
| 安装位置 | `/opt/rag-eval` |
| 定时任务 | `rag-eval.timer`，每日 02:30 CST |
| 首次跑批 | 门禁通过，指标与本机完全一致（index_version `2e382c34698a`） |

## DeepSeek 与 LLM 判官

DeepSeek 只提供对话模型，**没有 embedding 接口**，所以分工是：

| 环节 | 用什么 | 原因 |
|---|---|---|
| 向量化 | 本地 hashing（可换 OpenAI） | DeepSeek 没有 embedding 接口 |
| 生成 | 本地抽取式，可切 DeepSeek | 基线要确定性，换模型时要能对比 |
| 拒答判定 / claim 拆解 | DeepSeek 判官 | 语义判断，词汇信号做不到 |

判官层（`src/rag/judge.py`）有两个必要的工程约束：

- **缓存**：判定结果按 (模型, 问句, 资料) 哈希落盘，同一批数据反复跑不重复花钱；
- **熔断**：连不上时连续失败 3 次就停，按 `answerability.on_error` 兜底，绝不让整批跑批卡死。

所以无外网服务器的做法是：在本机预热判官缓存 → 连同缓存一起打进离线包 → 服务器上设 `judge.cache_only: true` 只读缓存。

接入层用本地假服务端做了集成测试（`tests/test_llm_judge.py`），验证了 HTTP 调用、JSON 提取、缓存命中不重复请求、熔断生效——不碰真实 API、不花钱。
