# ReviewProof 项目概览

用户选定一件商品后提问，系统只依据这件商品的买家评论回答：能答时逐条带引用作答；只有部分证据时说明哪部分能确认；评论没提到时明确说"评论无法确认"，不编造；答案下面另附一段标明"不是评论内容"的通用参考或查证建议。

本文汇总技术亮点、项目结构和使用指南。评测数字都取自 `reports/eval/*/metrics.json`，两个数字时前 dev 后 test，方括号是 95% 置信区间。详细内容见[技术报告](technical_report_cn.md)和[设计决策与问题复盘](design_and_lessons_cn.md)。

| 规模 | 数值 |
|---|---|
| 数据 | AmazonQA 验证集抽样：4 个品类、100 件商品、22,433 条评论、24,599 个评论块、767 道题 |
| 金标 | 200 题（dev、test 各 100），Claude 子 agent 预标、一位人工逐题审核，不是双人独立标注 |
| 测试 | Python 337 个，前端 57 个 |

## 1. 技术亮点

### 1.1 三态可答性，服务故障单独表示

每个回答输出 `answered / partial / unanswerable`，这是系统的核心指标。检索或模型调用失败时用 `service_status=failed` 单独表示，不能说成"评论无法确认"。这条规则贯穿检索、工作流、接口、前端徽标和评测分母，各层都有测试。

金标上，工作流答或拒的准确率为 0.850 [0.780, 0.910] / 0.830 [0.750, 0.900]。AmazonQA 自带分类器（B2）是 0.780 / 0.780；工作流的点估计更高，但区间重叠，不能说显著更好。

### 1.2 按评论量自适应的检索

评论总量不超过 6,000 token 的商品（100 件里 28 件）把全部评论交给模型，其余商品才走检索。大多数商品根本不需要检索，这个判断来自对数据的核实，不是默认加一层 RAG。

在同一批混合检索商品上，工作流和全量上下文（B1）分不出准确率高下（0.843 对 0.929、0.814 对 0.743，方向相反），而 B1 每题贵约 27 倍和 10 倍。

### 1.3 混合检索加重排

BGE-M3 向量和 BM25 各取前 50，经 RRF（k=60）融合，前 40 条交给 `bge-reranker-v2-m3` 重排，取 8 条作证据。检索都限定在单件商品内，向量用 pgvector 精确排序，不需要 ANN 索引和 OpenSearch。

金标上 nDCG@5 为 0.642 / 0.650，比 BM25 高 0.211 [0.135, 0.286] / 0.238 [0.148, 0.318]，比不重排的混合高 0.138 [0.068, 0.212] / 0.132 [0.074, 0.192]。任何一路失败都会记录状态、原因和耗时，另一路继续工作。

### 1.4 LangGraph 工作流

```
plan → retrieve → assess →（rewrite → 重新 assess，最多 1 次）→ generate → verify → finalize →（advise）
```

LangGraph 是唯一的编排路径。这里有条件分支、带预算的循环和逐节点 trace，节点跑完就能把进度推给前端。降级放在节点内部：规划失败时用规则继续，标为 `degraded`；检索、评估或生成失败时标为 `failed`，答案明说是服务出错。

`advise` 是定稿之后的通用参考：评论无法确认、部分能回答（只针对缺的部分）或问题超出范围时，单独调用一次模型，说明这类商品一般怎样，或建议去哪里确认。它看不到评论原文，不带引用，不改变判定，失败时只是不显示；评测时关闭，所以评测数字都不含它。

### 1.5 用代码约束模型输出

- **引用核验是纯代码：** 删掉不在本次候选里或属于别的商品的引用，没有引用的结论整条删除。这一步只核验引用 ID，还没有逐条核对结论是否真的被评论支持。
- **单一裁决者：** 三态只由证据评估决定，生成步骤不能改判。
- **是非题先承诺再作答：** `GeneratedYesNoAnswer` 的第一个字段 `verdict_question` 写明回答的是哪一问，最后才写结论；结论为 `unknown` 时整题降为不可答，保证 `unknown` 只和 unanswerable 一起出现。
- **回答语言由代码决定：** 关键表述用固定文案，不交给模型发挥。
- **所有 LLM 失败统一成 `LLMError`：** 包括输出不合 schema、缺少 API key，由节点统一降级。

### 1.6 核验之后才发布的流式接口

`POST /v1/qa` 用 SSE 实时推送每个节点的进度。答案在引用核验完成后才分块发出，不是逐 token 生成，否则用户会先看到未核验的引用。每次运行在发出答案前写入 `qa_runs`，可以按 `run_id` 回查候选、分数和节点决策；客户端中途断开，后台也会跑完并存档。

### 1.7 能推翻结论的评测体系

- **发现银标的偏差：** AmazonQA 自带的检索参考证据就是它自己的 BM25 前 10 条，用它比较检索方案，天然偏向 BM25。银标下 BM25 的 hit@5 最高，换成金标后结论反过来，混合加重排最好。
- **对照基线：** B0 不给证据、B1 全量上下文、B2 AmazonQA 分类器。
- **统计：** 指标附 bootstrap 95% 区间，方案比较用配对差值；服务故障不进任何可答性分母。
- **可复现：** 抽样清单记录数据文件的 SHA-256，每份报告的 `config.json` 记录代码版本、模型和参数；评测中断后重跑不重复花钱。
- **防泄漏：** 社区回答、片段和标签只存在 `questions` 表，有测试保证它们不会进入评论表、块表或提示词。

### 1.8 用评测做决定

- **误拒修复没有成立，全部回滚：** 三处改动让 dev 上误拒从 11 题降到 4 题，误答却从 4 题升到 11 题，准确率仍是 0.850，只是在两类错误之间换位置。
- **检查后决定不训练可答性模型：** 重排最高分和可答性有关（AUC 0.802 [0.689, 0.892]），但只看分数的准确率（0.830，阈值在同一批题上挑的）不如工作流，在工作流拒答的题里也分不开误拒和正确拒答（AUC 0.608 [0.364, 0.837]）。可答性继续由 LLM 根据证据判断。

各模块的选择、替代方案、依据和弱点汇总在复盘文档末尾的模块设计体检表。

### 1.9 工程化

- `docker compose up --build` 一条命令启动数据库、数据导入、接口和网页。
- 导入可重复运行，向量化按批提交，中断后接着算。
- 数据库测试拒绝连接名字不以 `_test` 结尾的库，防止清空正式数据。
- GitHub Actions 带 pgvector 服务容器运行测试。
- 成本和延迟：平均每题 3.14 次 Claude Haiku 5.5 调用，约 $0.0011；dev 上延迟 P50 14.5 秒、P95 27.1 秒（CPU 上运行重排）。

### 1.10 已知局限

- 误拒率 0.143 / 0.175，多数是金标为 partial 的题；是非题结论准确率只有 0.583 / 0.533（12 / 15 题）。
- 每个 split 约 100 题，区间较宽；每个系统每题只跑一次，LLM 输出有随机性。
- 引用正确率和结论支持率没有评测；中文提问能用，但没有单独题集。
- 全进程同时只跑一个问答，是原型阶段的明确取舍。

## 2. 项目结构

```text
ReviewProof/
├── reviewproof/                          # Python 主包
│   ├── config.py                         # 读取 REVIEWPROOF_* 配置
│   ├── db.py                             # 数据库连接和建表
│   ├── schema.sql                        # 表结构：商品、评论、评论块（1024 维向量）、题目、运行记录等
│   ├── embedding.py                      # BGE-M3 向量模型封装和设备选择
│   ├── llm.py                            # Claude 结构化输出，所有失败统一成 LLMError
│   ├── text.py                           # 唯一的文本规范化函数 normalize_text
│   ├── runs.py                           # 保存和查询 qa_runs 运行记录
│   ├── mcp_server.py                     # 只读 MCP 工具：list_products、search_reviews、get_review
│   ├── ingest/                           # 离线数据导入
│   │   ├── download.py                   # 下载 AmazonQA 数据文件并核对 SHA-256
│   │   ├── sampling.py                   # 生成固定抽样清单
│   │   ├── amazonqa.py                   # 读取商品、评论和题目，生成系统评论 ID
│   │   ├── chunking.py                   # 按句切分评论并保存偏移
│   │   ├── loader.py                     # 幂等写库，文本不变时保留向量
│   │   ├── embed.py                      # 可断点续跑的向量化
│   │   ├── titles.py                     # 从 2023、2014 两份 Amazon 元数据补商品标题
│   │   └── cli.py                        # 导入命令：核对、写库、向量化、校验
│   ├── retrieval/                        # 单商品检索
│   │   ├── retriever.py                  # 策略选择、RRF 融合、重排和状态汇总
│   │   ├── dense.py                      # pgvector 精确向量检索
│   │   ├── bm25.py                       # 按商品缓存的 BM25
│   │   ├── rerank.py                     # bge-reranker-v2-m3 重排
│   │   ├── types.py                      # 候选、通道状态、检索结果等类型
│   │   └── cli.py                        # 命令行查看排序结果
│   ├── workflow/                         # LangGraph 问答工作流
│   │   ├── graph.py                      # 状态图，唯一的编排路径
│   │   ├── nodes.py                      # plan、retrieve、assess、rewrite、generate 等节点
│   │   ├── answer.py                     # 纯代码核验引用、确定最终状态、渲染固定文案
│   │   ├── prompts.py                    # 三个 LLM 步骤的提示词
│   │   ├── schemas.py                    # 每一步要求模型返回的结构化输出
│   │   ├── state.py                      # 工作流状态和依赖
│   │   └── cli.py                        # 命令行提问
│   ├── api/                              # FastAPI 服务
│   │   ├── app.py                        # 入口：健康检查、商品、证据、运行记录、/v1/qa，并提供前端页面
│   │   ├── qa.py                         # 问答的 SSE 事件流
│   │   ├── services.py                   # 启动时加载一次模型和 LLM 客户端
│   │   ├── queries.py                    # 接口和 MCP 共用的只读查询
│   │   └── annotation.py                 # 本地标注审核接口，默认关闭
│   ├── eval/                             # 评测
│   │   ├── cli.py                        # 评测入口
│   │   ├── dataset.py                    # 读取和分层抽样评测题
│   │   ├── alignment.py                  # 把 AmazonQA 片段对齐到评论块，生成银标参考证据
│   │   ├── retrieval_eval.py             # 比较 BM25、向量、混合、混合加重排四种检索方案
│   │   ├── workflow_eval.py              # 运行工作流和 B0、B1 基线
│   │   ├── baselines.py                  # 基线 B0：不给证据
│   │   ├── gold.py                       # 读取并校验金标、计算一致率
│   │   ├── metrics.py                    # 排序指标、bootstrap 区间、配对差值
│   │   ├── report.py                     # 生成 metrics.json 和中文 report.md
│   │   └── rerank_check.py               # 第 6 阶段的重排分数检查
│   └── annotate/                         # 金标工具
│       ├── pool.py                       # 抽题并固定候选池
│       ├── store.py                      # 保存预标、盲标和终审标注
│       └── cli.py                        # pick、packet、search、import-drafts、status、export
├── frontend/                             # Vue 3 + Vite + TypeScript 前端
│   └── src/
│       ├── App.vue                       # 问答主页面
│       ├── api/                          # 接口客户端和 SSE 解析
│       ├── composables/useAsk.ts         # 提问状态管理
│       ├── components/                   # 商品选择、提问、进度、答案、引用、运行详情等组件
│       └── annotate/                     # 金标审核页面
├── tests/                                # Python 测试（unittest）
│   └── support.py                        # 共享的测试工具和假模型
├── samples/
│   ├── manifest_v1.json                  # 固定抽样清单，含数据文件 SHA-256
│   └── manifest_v1.titles.json           # 100 件商品的标题、品牌和来源
├── annotations/gold_v1/labels.jsonl      # 导出的 200 题金标
├── reports/eval/                         # 评测报告
│   ├── pilot-dev/、pilot-test/           # 银标试跑，只作记录
│   ├── gold-dev/、gold-test/             # 金标评测
│   ├── gold-dev-r2/、gold-dev-r3/        # 误拒修复实验，已回滚
│   └── rerank-check-dev/                 # 第 6 阶段的重排分数检查
├── docs/
│   ├── product_positioning_cn.md         # 项目定位：做什么、不做什么
│   ├── roadmap_cn.md                     # 技术决策和阶段划分
│   ├── technical_report_cn.md            # 技术报告
│   ├── design_and_lessons_cn.md          # 设计决策、问题复盘和模块设计体检表
│   ├── annotation_guidelines_cn.md       # 金标标注规范
│   └── project_overview_cn.md            # 本文
├── docker/initdb/01-test-db.sql          # 创建测试数据库
├── .github/workflows/tests.yml           # GitHub Actions 测试
├── docker-compose.yml                    # Docker 全栈编排：db、ingest、api
├── Dockerfile                            # 构建前端并打包 Python 应用
├── requirements.txt                      # Python 依赖
├── .env.example                          # 配置模板
├── .env                                  # 本地配置，需自行创建；不要提交密钥
└── data/                                 # AmazonQA 数据文件，自动下载，不进 git
```

## 3. 使用指南

### 3.1 环境要求

| 用途 | 需要 |
|---|---|
| 用 Docker 启动 | Docker，Compose v2.24 以上 |
| 本地开发 | Python 3.10+、Docker（运行数据库）；构建前端另需 Node 20 以上 |
| 提问 | `ANTHROPIC_API_KEY`；没有它也能启动和浏览，提问会显示"服务出错" |

### 3.2 用 Docker 一键启动

```bash
cp .env.example .env              # 可选：填入 ANTHROPIC_API_KEY 才能提问
docker compose up --build
```

完成后打开 http://127.0.0.1:8000/ 。第一次运行要下载约 1.5 GB 数据和两个模型（各约 2.3 GB），并计算 24,599 个评论块的向量。之后再启动时，导入步骤约 7 秒，网页约半分钟内可用。

Apple 芯片的 Mac 上，Docker 里算向量要 4 个多小时，建议先用本机 Python 导入（约 40 分钟）：

```bash
docker compose up -d db
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m reviewproof.ingest.download --manifest samples/manifest_v1.json
python -m reviewproof.ingest --manifest samples/manifest_v1.json
docker compose up --build
```

### 3.3 本地开发启动

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # 填入 ANTHROPIC_API_KEY
docker compose up -d db           # Postgres 16 + pgvector，端口 54329
python -m reviewproof.ingest.download --manifest samples/manifest_v1.json
python -m reviewproof.ingest --manifest samples/manifest_v1.json
cd frontend && npm ci && npm run build && cd ..
python -m reviewproof.api         # 打开 http://127.0.0.1:8000/
```

数据和向量已经导入过时，只需要 `docker compose up -d db` 和 `python -m reviewproof.api`。

开发前端时另开一个终端运行 `cd frontend && npm run dev`，打开 http://127.0.0.1:5173/ ，请求会转发到 8000 端口。

### 3.4 使用网页

1. 选择品类和商品。下拉框显示商品标题和评论数，选中后显示完整标题、ASIN 和检索方式。
2. 选一道示例问题，或自己输入问题，中英文都可以。
3. 查看每个节点的实时进度。
4. 查看答案状态：能回答、部分能回答、评论无法确认；是非题另给结论。评论答不了的部分，答案下面的灰色区块是通用参考，不是评论内容。
5. 点开引用查看原评论，需要时查看本次运行详情。

可以试试这些问题：

| 商品 | 问题 | 预期 |
|---|---|---|
| B00007E7C8 | Does it have volume control? | 走混合检索，带引用作答 |
| B00007E7C8 | 音量可以调节吗？ | 自动译成英文检索，用中文回答 |
| 任意商品 | How much does it cost? | 判为不在服务范围，并建议去商品页或问卖家 |

### 3.5 命令行

```bash
python -m reviewproof.retrieval --qid 17020                                    # 查看某道题的检索结果
python -m reviewproof.retrieval --product B00007E7C8 --query "volume control?" # 指定商品和查询
python -m reviewproof.workflow --qid 17020                                     # 回答某道题，需要 API key
python -m reviewproof.workflow --product B00007E7C8 --question "音量可以调节吗？" --json
```

### 3.6 接口和 MCP

| 接口 | 说明 |
|---|---|
| `GET /healthz` | 健康检查 |
| `GET /v1/products?category=&split=` | 商品列表 |
| `GET /v1/products/{id}/questions` | 商品的示例问题 |
| `GET /v1/evidence/{review_id}` | 评论原文 |
| `GET /v1/runs/{run_id}` | 查询运行记录 |
| `POST /v1/qa` | 流式问答（SSE），事件为 `run_started`、`progress`、`answer_chunk`、`citations`、`done`、`error` |

服务故障也以 `done` 结束并返回 HTTP 200；只有输入错误返回 4xx（商品不存在 404，问题为空 422）。

只读 MCP 服务提供 `list_products`、`search_reviews`、`get_review`，不提供问答工具：

```bash
python -m reviewproof.mcp_server    # 默认 stdio；用 REVIEWPROOF_MCP_TRANSPORT 切换为 sse 或 streamable-http
```

### 3.7 评测

```bash
python -m reviewproof.eval --split test --name my-run --retrieval-only   # 只做检索评测，不调用 Claude
python -m reviewproof.eval --split dev --name gold-dev --labels annotations/gold_v1/labels.jsonl   # 金标评测，会调用 Claude
python -m reviewproof.eval.rerank_check --labels annotations/gold_v1/labels.jsonl --split dev --workflow reports/eval/gold-dev/per_question.jsonl --name rerank-check-dev   # 重排分数检查，不调用 Claude
```

结果写到 `reports/eval/<name>/`：`report.md` 是中文报告，`metrics.json` 是全部指标，`per_question.jsonl` 是逐题结果，`config.json` 记录代码版本、模型和参数。金标评测中断后，重跑同一条命令会跳过已完成的题，重试出过服务故障的题。

### 3.8 测试

```bash
python -m unittest discover -s tests -b
REVIEWPROOF_TEST_DATABASE_URL=postgresql://reviewproof:reviewproof@localhost:54329/reviewproof_test python -m unittest discover -s tests -b
cd frontend && npm run typecheck && npm test
```

第一条只跑不需要数据库的测试；第二条连同数据库测试一起跑，要先运行 `docker compose up -d db`，并且只会清空名字以 `_test` 结尾的测试库。

### 3.9 常用配置

配置模板在 `.env.example`。常用项：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `ANTHROPIC_API_KEY` | 无 | Claude API key |
| `REVIEWPROOF_DATABASE_URL` | `postgresql://reviewproof:reviewproof@localhost:54329/reviewproof` | 数据库连接 |
| `REVIEWPROOF_EMBEDDING_DEVICE` | 有 CUDA 用 CUDA，否则 CPU | 向量和重排使用的设备 |
| `REVIEWPROOF_FULL_CONTEXT_MAX_TOKENS` | `6000` | 全量上下文阈值 |
| `REVIEWPROOF_MAX_REWRITES` | `1` | 改写重检索次数上限（0–2） |
| `REVIEWPROOF_LLM_MODEL` | `claude-haiku-5-5` | 生成模型 |
| `REVIEWPROOF_API_PORT`、`REVIEWPROOF_DB_PORT` | `8000`、`54329` | Docker 映射到本机的端口 |

### 3.10 常见问题

- **页面显示接口或数据库连不上：** 先确认 `docker compose up -d db` 已运行，再启动接口。
- **提问显示"服务出错"：** 通常是没有设置 `ANTHROPIC_API_KEY`，或模型调用失败。这是服务故障，不代表评论里没有答案。
- **数据库测试全部报 `Connection refused`：** 数据库容器没有运行。
- **导入以退出码 2 结束：** 数据文件的 SHA-256 和抽样清单不一致，重新下载即可。
- **不要直接打印 `docker compose config`：** 它会把 `.env` 里的 API key 展开输出。
