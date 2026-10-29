# ReviewProof

购前单商品评论问答：选定一件商品，用这件商品的真实评论回答购买前的问题。每个回答标明"能回答 / 部分能回答 / 评论里找不到"，并给出引用的评论原文。数据来自 [AmazonQA](https://github.com/amazonqa/amazonqa)。

- 做什么、不做什么：[项目定位](docs/product_positioning_cn.md)
- 怎么做、分几个阶段：[改动方案](docs/roadmap_cn.md)
- 系统和评测结果的完整说明：[技术报告](docs/technical_report_cn.md)；设计取舍和遇到的问题：[设计决策与问题复盘](docs/design_and_lessons_cn.md)

## 用 Docker 一键启动

只需要 Docker（Compose v2.24 以上）。在仓库根目录运行：

```bash
cp .env.example .env              # 可选：填入 ANTHROPIC_API_KEY 才能提问
docker compose up --build
```

依次启动三个服务，完成后打开 http://127.0.0.1:8000/ ：

1. `db`：PostgreSQL 16 + pgvector。
2. `ingest`：从 AmazonQA 官方 S3 下载两份数据文件（约 1.5 GB，按 `samples/manifest_v1.json` 核对 SHA-256；`data/` 里已有就跳过），导入抽样的 100 件商品，计算 BGE-M3 向量，最后校验。可以重复运行，中断后再启动会接着算剩下的向量。
3. `api`：等 `ingest` 成功后启动接口和网页。

需要知道的：

- **第一次很慢。** 镜像构建约 1 分钟（本机实测，取决于网速）。除了数据，还要下载两个模型（`BAAI/bge-m3` 和 `BAAI/bge-reranker-v2-m3`，各约 2.3 GB，存进 `reviewproof-models` 卷），并在 CPU 上计算 24,599 个评论块的向量。在 Apple 芯片的 Mac 上，Docker 里实测每分钟 85–96 块，算完要 4 个多小时，因为 Linux 虚拟机里的 PyTorch 用不上 macOS 的加速库；同一台机器上直接用本机 Python 导入约 40 分钟。所以 Mac 上建议先用本机 Python 导入，再启动全栈（见下面的命令）。其他平台的速度没有测过。
- **之后再启动**时，`ingest` 会重新核对数据、确认向量都在（实测约 7 秒），网页约半分钟内可用。
- **没有 API key 也能启动：** 可以浏览商品、查看评论，提问会显示"服务出错"，不会说成"评论无法确认"。
- **速度和内存：** Docker 里每次检索（含重排）实测 11–16 秒；本机 Python 上重排约 5 秒。接口进程加载两个模型，本机实测约 2.4 GB；本机给 Docker 分配 5 GB 内存时整套运行正常。
- **端口：** 只监听本机。网页默认 8000、数据库默认 54329，可以用 `REVIEWPROOF_API_PORT`、`REVIEWPROOF_DB_PORT` 改。
- Linux 上 `data/` 由容器创建时属于 root。

Mac 上更快的做法（需要 Python 3.10+）：

```bash
docker compose up -d db
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m reviewproof.ingest.download --manifest samples/manifest_v1.json
python -m reviewproof.ingest --manifest samples/manifest_v1.json    # 约 40 分钟
docker compose up --build                                          # ingest 发现向量都在，很快结束
```

## 当前进度

- 第 1 阶段（数据底座）：从 AmazonQA 抽样 100 件商品（4 个品类），共 22,433 条评论、24,599 个评论块和 767 道评测题，存入 PostgreSQL + pgvector，每个块都有 BGE-M3 向量。
- 第 2 阶段（混合检索）：给定商品和问题，评论少的商品直接返回全部评论块，其余商品用向量检索和 BM25 两路召回、RRF 融合、BGE 重排，取 8 条证据。任一路出故障时另一路继续，并记录实际用了哪些通道。
- 第 3 阶段（问答工作流）：LangGraph 把规划、检索、证据评估、必要时改写重检索、带引用生成和引用核验串起来，输出"能回答 / 部分能回答 / 评论无法确认"三态和是非题结论；服务出错时单独说明，不报成"评论无法确认"。生成用 Claude Haiku 5.5。
- 第 4 阶段（接口和界面）：FastAPI 接口（流式问答 SSE、商品、示例问题、评论原文、运行记录）、Vue 3 前端页面、只读 MCP 检索服务；每次运行存进 PostgreSQL，可按 `run_id` 回查。
- 第 5a 阶段（银标评测，小规模试跑）：用 AmazonQA 自带的银标评测检索（BM25、向量、混合、混合加重排四种方案）和可答性（完整工作流对比 B0 不给证据、B1 全量上下文），每个指标带 bootstrap 95% 置信区间。dev 和 test 各抽 50 题试跑，报告见 [pilot-dev](reports/eval/pilot-dev/report.md) 和 [pilot-test](reports/eval/pilot-test/report.md)。银标不是人工金标：检索参考证据就是 AmazonQA 自己的 BM25 前 10 条，天然偏向 BM25；所以这两份报告只是试跑记录，不据此下结论，结论留到 5b 阶段的人工金标评测。
- 第 5b 阶段（人工金标评测）：dev 和 test 各抽 100 题，由 Claude 子 agent 按[标注规范](docs/annotation_guidelines_cn.md)预标，再由一位人工逐题审核，其中 40 题先盲标；每个候选块标 0/1/2 级证据。报告见 [gold-dev](reports/eval/gold-dev/report.md) 和 [gold-test](reports/eval/gold-test/report.md)。金标下的主要结果（均值 [95% 置信区间]）：
  - 检索：混合检索加重排在两个 split 上 nDCG@5 都最高；减去 BM25 的配对差值，dev 为 0.211 [0.135, 0.286]，test 为 0.238 [0.148, 0.318]。银标试跑里重排看起来不如 BM25，原因是银标的参考证据就是 AmazonQA 自己的 BM25 结果。
  - 可答性（答或拒是否正确）：完整工作流 dev 0.850 [0.780, 0.910]、test 0.830 [0.750, 0.900]；不给证据的 B0 为 0.300 和 0.250；AmazonQA 自带分类器（B2）都是 0.780。和全量上下文 B1 在同一批题上比，dev 上 B1 的点估计更高（0.929 对 0.843），test 上工作流更高（0.814 对 0.743），区间都重叠，分不出高下；工作流每题成本约为 B1 的 1/27（dev）和 1/10（test）。
  - 主要错误是误拒（金标可答、系统拒答），误拒率 dev 0.143、test 0.175；是非题结论准确率只有 0.583（12 题）和 0.533（15 题）。
  - 针对误拒试过三处提示词和规则修改，dev 上没有成立：准确率仍是 0.850，只是把误拒换成了误答（回滚一处后误拒 11 → 8、误答 4 → 7）。已全部回滚，test 没有复跑。报告见 [gold-dev-r2](reports/eval/gold-dev-r2/report.md) 和 [gold-dev-r3](reports/eval/gold-dev-r3/report.md)，原因见[复盘文档](docs/design_and_lessons_cn.md) 9.5 节。
  - 局限：只有一位人工，盲标一致率是人和子 agent 之间的；每个 split 约 100 题，区间较宽；test 上的一道试跑错题曾用于修改提示词（那道题不在金标里）。
- 第 7 阶段（收尾）：`docker compose up --build` 一条命令启动全栈（下载并核对数据、导入、向量化、接口和网页），在干净克隆、没有 `.env` 的环境里验证过（模型用的是本机已下载的缓存；Docker 里向量算得太慢，只在容器里算了前 736 个块，其余按块编号从开发库复制，再由导入步骤校验）；写了[技术报告](docs/technical_report_cn.md)。

## 本地开发

需要 Python 3.10+ 和 Docker。

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # 填入 ANTHROPIC_API_KEY（第 3 阶段起用于生成回答）
docker compose up -d db           # 本地 Postgres 16 + pgvector，端口 54329
```

下载 AmazonQA 验证集的两份文件到 `data/`（已有且 SHA-256 一致就跳过），然后导入：

```bash
python -m reviewproof.ingest.download --manifest samples/manifest_v1.json
python -m reviewproof.ingest --manifest samples/manifest_v1.json
```

命令会先核对数据文件的 SHA-256，再写入商品、评论、评论块和题目，最后计算向量并校验。可以重复运行：已经写入的数据不会重复，中途中断后再运行会接着计算剩下的向量。首次运行会从 Hugging Face 下载 `BAAI/bge-m3`（约 2.3 GB）。

查看某道题的检索结果（第一次运行会下载重排模型 `BAAI/bge-reranker-v2-m3`，约 2.3 GB；加 `--no-rerank` 可以跳过重排）：

```bash
python -m reviewproof.retrieval --qid 17020
python -m reviewproof.retrieval --product B00007E7C8 --query "Does it have volume control?"
```

提问（需要在 `.env` 填 `ANTHROPIC_API_KEY`，每题约 3–4 次 Claude Haiku 5.5 调用）：

```bash
python -m reviewproof.workflow --qid 17020
python -m reviewproof.workflow --product B00007E7C8 --question "音量可以调节吗？"
```

## 启动网页

```bash
cd frontend && npm ci && npm run build && cd ..
python -m reviewproof.api        # 启动时加载模型约 15 秒，然后打开 http://127.0.0.1:8000/
```

开发前端时另开一个终端运行 `cd frontend && npm run dev`，打开 http://127.0.0.1:5173/（请求会转发到 8000 端口的接口）。进度是实时推送的；答案在引用核验完成后分段显示，不是逐字生成。

## 评测

```bash
python -m reviewproof.eval --split dev --name pilot-dev                    # 检索评测（整个 split）+ 工作流和两个基线（抽 50 题），会调用 Claude
python -m reviewproof.eval --split test --name my-run --retrieval-only     # 只做检索评测，不调用 Claude
python -m reviewproof.eval --split dev --name gold-dev --labels annotations/gold_v1/labels.jsonl   # 按金标评测这个 split 的全部金标题，会调用 Claude
```

金标的标注流程：`python -m reviewproof.annotate pick` 抽题并固定候选池，子 agent 预标后用 `import-drafts` 导入，人工在 `REVIEWPROOF_ANNOTATION=1 python -m reviewproof.api` 启动的页面 `http://127.0.0.1:8000/#/annotate?set=gold_v1` 上逐题审核，最后 `python -m reviewproof.annotate export --set gold_v1` 导出到 `annotations/gold_v1/labels.jsonl`。

结果写到 `reports/eval/<name>/`：`report.md` 是中文报告，`metrics.json` 是全部指标，`per_question.jsonl` 和 `retrieval.jsonl` 是逐题结果，`config.json` 记录代码版本、模型和参数。中断后重跑同一条命令会跳过已完成的题，重试出过服务故障的题。

## 测试

```bash
python -m unittest discover -s tests -b
REVIEWPROOF_TEST_DATABASE_URL=postgresql://reviewproof:reviewproof@localhost:54329/reviewproof_test python -m unittest discover -s tests -b
```

第一条只跑不需要数据库的测试；第二条连同数据库测试一起跑，只会清空名字以 `_test` 结尾的测试库。前端测试：`cd frontend && npm run typecheck && npm test`。

## 数据来源

数据来自 AmazonQA（Mansi Gupta, Nitish Kulkarni, Raghuveer Chanda, Anirudha Rayasam, Zachary C. Lipton. *AmazonQA: A Review-Based Question Answering Task*. IJCAI 2019），从官方 S3 下载：`https://amazon-qa.s3-us-west-2.amazonaws.com/val-qar_products.jsonl` 和 `val-qar.jsonl`。AmazonQA 没有声明数据许可，所以本仓库不分发数据文件或数据库快照，只保存抽样清单（商品 ASIN 和文件的 SHA-256）。评测报告和金标里有题目文本和评论摘录。评论 ID（如 `B00007E7C8-r00012`）由本系统生成，不是 Amazon 的评论 ID。
