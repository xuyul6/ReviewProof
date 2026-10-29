# ReviewProof 改动方案

日期：2026-10-27。

[项目定位](product_positioning_cn.md) 决定了"做什么"。本文决定"怎么改、分几个阶段、每个阶段交付什么"。每个阶段开始前，会在 `docs/superpowers/plans/` 写一份逐步实施计划；第 1 阶段的计划已经写好：[2026-10-27-phase-1-data-foundation.md](superpowers/plans/2026-10-27-phase-1-data-foundation.md)。

## 1. 目标架构

```
离线
  val-qar_products.jsonl ─ 抽样清单 ─ 读取与合并 ─ 系统评论 ID ─ 句级切分
     ├─ BGE-M3 向量化 ──────→ PostgreSQL + pgvector（chunks 表，按商品过滤）
     └─ 问题 / 社区回答 / 可答性银标 → questions 表（只用于评测，不进检索）
  （第 6 阶段原计划：train-qar.jsonl 银标 → 微调可答性模型 → 用人工金标校准阈值；检查后决定不做，见第 3 节）

在线（FastAPI + LangGraph，唯一主链路）
  校验商品 → 意图路由 → 查询规划（改写；中文问题翻成英文关键词）
    → 选择策略：评论少 → 全部评论放进上下文
                评论多 → 稠密 ∥ BM25 → RRF 融合 → BGE 重排
    → 证据评估（可答 / 部分可答 / 不可答）
         证据不足且还有预算 → 改写后重检索（最多 1 次）
    → 带引用生成（结构化 JSON）→ 引用核验
    → SSE：进度事件 + 核验后分块发布答案
    → 运行记录写入 PostgreSQL

评测
  固定清单与题集 → 检索 / 可答性 / 引用 / 长上下文对照 / 成本与延迟
    → 报告（95% 置信区间 + 失败案例）
```

## 2. 关键技术决定

| 事项 | 决定 | 理由 | 取代的现有做法 |
|---|---|---|---|
| 代码组织 | 独立仓库，代码全部在 `reviewproof/` 包；环境变量前缀 `REVIEWPROOF_` | 旧系统留在旧仓库，不值得先重命名再重写 | 在旧仓库里沿用 `bridge/` 包、`BRIDGE_*` 变量 |
| 数据库 | PostgreSQL 16 + pgvector，本地用 Docker Compose（端口 54329）；Python 用 psycopg 3 直接写 SQL | 需要按商品过滤、事务和关联查询，直接写 SQL 比 REST/RPC 简单；Supabase 也提供 Postgres 连接串，仍然可以用 | Supabase REST + `match_file_items_gemini` RPC |
| 语料 | AmazonQA `val-qar_products.jsonl` 按清单抽样 100 件商品 | 完整评论、与问题同源，见定位文档 5.3 节 | Reviews 2023 拼接文本、`file_id` |
| 切分 | 评论不超过 1,200 字符时整条保留；更长的按句子打包到约 800 字符，相邻块重叠一句 | 评论中位数 238 字符，大多不用切；块要能放进重排模型 512 token 的输入 | 整条评论 |
| 向量模型 | BGE-M3，1024 维，本地运行 | 免费、可复现、多语言（中文问题检索英文评论）；评测要大量跑查询，不受 API 限流影响 | Gemini 3072 维 |
| 向量检索 | 按商品精确排序，不建 ANN 索引 | 单件商品最多约 9,000 条评论，精确排序很快；先过滤再 ANN 会损失召回 | 全库 RPC |
| 稀疏检索 | `rank_bm25`，按商品建索引并缓存在内存 | 有了商品过滤，单商品规模很小；仍是真正的 BM25 | 每次请求拉前 5,000 行；目标架构里的 OpenSearch |
| 融合与重排 | RRF（k=60）→ `bge-reranker-v2-m3` 重排前 40 条 → 取 8 条作证据 | RRF 不受两路分数量纲影响；重排模型多语言 | 0.65/0.35 加权、`bge-reranker-base` |
| 自适应策略 | 商品评论总量不超过阈值时全部放进上下文，否则走混合检索；初始阈值 6,000 tokens，第 5 阶段在 dev 集上调 | 一半商品的评论不到约 3,000 tokens，这时检索没有意义 | 无 |
| 生成 | Claude Haiku 5.5（`claude-haiku-5-5`），用 Anthropic 官方 Python SDK，输出结构化 JSON：结论、引用 ID、未知项、是非结论 | Claude 最便宜的模型，评测要跑大量题目；结构化输出才能核验引用 | Gemini 2.5 Flash；纯文本回答 |
| 可答性 | LLM 证据评估是唯一裁决者；第 6 阶段检查过重排模型的分数，不如它，所以没有训练单独的模型 | 金标 dev 上重排最高分分不开误拒和正确拒答（AUC 0.608 [0.364, 0.837]）；用银标训练，上限是 AmazonQA 自己的分类器（金标准确率 0.780，低于工作流） | 数量与词重合启发式；用银标微调交叉编码器 |
| 编排 | LangGraph 是唯一主链路；在节点内降级 | 去掉"整图失败 → 手写工作流 → 直连推理"三套并行实现 | `BRIDGE_USE_LANGGRAPH` 开关 + 三层回退 |
| 流式 | SSE 事件：`run_started / progress / answer_chunk / citations / done / error`；答案核验后再分块发布 | 引用要先核验再给用户；如实说明这不是逐 token 生成 | OpenAI 风格 chunk |
| 运行记录 | PostgreSQL `qa_runs` 表 | 与数据同库，便于评测回查 | SQLite、打印日志 |
| 测试 | 继续用 `unittest`；纯逻辑测试离线运行；设置 `REVIEWPROOF_TEST_DATABASE_URL` 后才运行数据库测试；GitHub Actions 带 pgvector 服务容器 | 沿用现有习惯；CI 也能测数据库 | 无 CI |
| 状态命名 | `answered / partial / unanswerable`；服务故障单独用 `service_status` 表示 | 沿用小样本代码的命名 | 目标架构里的 `full / partial / none` |

## 3. 阶段划分

| 阶段 | 目标 | 主要改动 | 完成标准 |
|---|---|---|---|
| **1 数据底座** | 抽样的 100 件商品进入数据库，带向量 | `reviewproof/` 骨架、配置、Docker Compose、CI；抽样清单 `samples/manifest_v1.json`；AmazonQA 读取与合并；切分；建表与写库；BGE-M3 向量化；导入命令 | 一条命令完成导入；每个块都有向量；重复导入不产生重复数据；问题、回答、标签只在 `questions` 表 |
| **2 混合检索** | 给定商品和问题，返回带分数的证据候选 | `reviewproof/retrieval/`：稠密、BM25、RRF、重排、策略选择、通道状态；命令行查看排序结果 | 两路结果都只来自该商品；单路故障时另一路继续，并记录降级；评论少的商品走全量上下文 |
| **3 LangGraph 工作流** | 从问题到带引用的三态回答 | `reviewproof/workflow/`：状态、节点、图；`reviewproof/llm.py`（参照旧文件夹里已换成 Claude 的 `bridge/core/llm_client.py` 重写） | 输出三态之一；所有引用都来自本商品的候选；改写最多 1 次；服务故障和"评论无法确认"分开；每个分支都有离线测试 |
| **4 接口和界面** | 能演示的完整系统 | `reviewproof/api/`：`GET /v1/products`、`POST /v1/qa`（SSE）、`GET /v1/runs/{id}`、`GET /v1/evidence/{review_id}`；`qa_runs` 表；Vue 3 前端（`frontend/`，由 FastAPI 同源提供）；MCP Server 用新检索 | 页面上能选商品、提问、查看引用和运行详情；测试全部通过 |
| **5 评测体系** | 用数字说明系统好不好 | `reviewproof/eval/`：题集、片段到评论的对齐（生成候选 qrels）、人工标注导出导入、评测运行器（BM25、稠密、混合、+重排、全量上下文、完整工作流、无检索 B0）、指标与报告；在 dev 集调自适应阈值 | 同一数据、同一命令可以复跑；报告带置信区间和失败案例；**需要人工标注约 200 题** |
| **6 可答性模型（检查后不训练）** | 判断值不值得训练单独的可答性模型 | 原计划是下载 `train-qar.jsonl`、用银标微调交叉编码器、在金标 dev 上校准阈值。先做了一个不训练、不调 Claude 的检查：`reviewproof/eval/rerank_check.py`，看重排最高分能不能区分可答和不可答，报告 `reports/eval/rerank-check-dev` | 检查结论和不训练的理由写进技术报告 6.4 节和复盘文档第 10 节；微调不做 |
| **7 收尾** | 别人能一键跑起来 | Docker Compose 启动全栈；README、技术报告；决定是否公开仓库 | 全新环境按 README 能跑通 |

阶段之间有依赖：1 → 2 → 3 → 4；第 5 阶段依赖第 3 阶段（不依赖界面）；第 6 阶段依赖第 5 阶段的金标。

## 4. 旧代码

2026-10-27 起，本仓库只放新链路。旧系统（`bridge/`、`app/`、Supabase + Gemini 检索、小样本演示、旧报告）留在另一个未公开的旧仓库和本地的旧文件夹里，不再搬进来。下面这些在后续阶段可以去旧仓库或旧文件夹参考，但都是重写，不直接复制：

| 旧代码文件 | 参考它的阶段 |
|---|---|
| `bridge/core/llm_client.py`（Claude Haiku 5.5 版本只在旧文件夹的 `phase-1-data` 分支上，没有推送到 GitHub） | 第 3 阶段写 `reviewproof/llm.py` |
| `bridge/demo_qa.py`（逐条引用、三态答案、运行记录） | 第 3、4 阶段 |
| `bridge/graph/*`（旧 LangGraph 工作流和多级降级） | 第 3 阶段 |
| `bridge/mcp_server.py`、`app/api/main.py` | 第 4 阶段 |
| `scripts/evaluate_retrieval.py`（检索指标计算） | 第 5 阶段 |

## 5. 推迟或不做

- **商品名称：** 数据里没有。2026-10-29 按 ASIN 补上了全部 100 件商品的标题（`samples/manifest_v1.titles.json`）：50 件来自 Amazon Reviews 2023 元数据，其余 50 件来自 2014 年的 Amazon 商品数据。
- **多轮追问、中文提问：** 第 3 阶段在查询规划节点里保留中译英；评测在第 5 阶段用单独的小题集，不混入主指标。
- **相似历史问答：** 不排期。
- **OpenSearch、HNSW 索引：** 不做，除非评测数据证明需要。
