# 第 7 阶段：收尾 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 别人在全新环境里只装 Docker，按 README 一条命令就能把数据库、导入和网页跑起来；README 和技术报告说清楚系统和评测结果；最后做公开仓库前的检查，交给用户决定。

**Architecture:** 一个镜像两种用法：`ingest` 服务先从 AmazonQA 官方 S3 下载两份 val 文件（按清单 SHA-256 校验），再执行现有导入命令；`api` 服务等 `ingest` 成功后启动，同时提供前端。模型缓存放在命名卷里，数据目录绑定到仓库的 `data/`。不分发任何数据库快照。

**Tech Stack:** Docker Compose v2、`python:3.12-slim` + CPU 版 torch、`node:24-slim`（构建前端）、`pgvector/pgvector:pg16`、Python 标准库 `urllib`（下载）、`unittest`。

**Spec:** `docs/roadmap_cn.md` 第 3 节第 7 行（"Docker Compose 启动全栈；README、技术报告；决定是否公开仓库"，完成标准"全新环境按 README 能跑通"），加上用户 2026-10-29 的四个决定（见 Global Constraints）。

## Global Constraints

- 数据：全新环境从 `https://amazon-qa.s3-us-west-2.amazonaws.com/` 下载 `val-qar_products.jsonl`（731,873,701 字节）和 `val-qar.jsonl`（747,051,157 字节），SHA-256 必须等于 `samples/manifest_v1.json` 的 `source`；**不发布数据库快照或任何数据副本**（AmazonQA 没有声明许可）。
- 文档语言：README 和技术报告都用中文，和现有文档一致。
- 公开仓库：只做检查、列出发现，**由用户决定**；不改仓库可见性，不改写 git 历史。
- `.env`（含 `ANTHROPIC_API_KEY`）绝不能进镜像、日志或提交。
- 已有的开发用法不变：`docker compose up -d db` 只起数据库，端口默认仍是 54329，测试库仍由 `docker/initdb` 创建。
- 真实 Claude 调用、模型下载、合并、推送都要先征得用户同意。本计划的验证不调用 Claude，不重新下载模型（绑定本机已有的 Hugging Face 缓存）。
- 文档里的数字只能来自 `reports/eval/*/metrics.json` 或实际测得的结果；银标、金标、系统评论 ID 的口径照旧。

## Review Focus

1. **没有 API key 就启动**（全新克隆没有 `.env`）：整套服务照常起来，页面能浏览商品；提问以"服务出错"结束，而不是"评论无法确认"。Task 2 的全新环境验证专门走这条路。
2. **下载中途断开**：不能留下一个半截文件被当成完整数据。Task 1 先写 `.part` 再改名，SHA 不符时什么都不留（测试 `test_sha_mismatch_after_download_leaves_no_file`）。
3. **本地已有一份不同的数据文件**：不能被悄悄覆盖。Task 1 拒绝并保留原文件（测试 `test_existing_file_with_wrong_sha_is_refused_and_kept`）。
4. **`.env` 被打进镜像**：Task 2 的 `.dockerignore` 测试，加上验证时在镜像里列 `/app` 确认没有 `.env`。
5. **Docker 内存不够**，API 被杀掉（退出码 137）却没有提示：Task 2 用 `docker stats` 实测内存，README 写明要求。

---

### Task 1: 数据下载命令

**Files:**
- Create: `reviewproof/ingest/download.py`
- Test: `tests/test_download.py`

**Interfaces:**
- Consumes: `reviewproof.ingest.amazonqa.file_sha256(path: Path) -> str`
- Produces:
  - `AMAZONQA_BASE_URL = "https://amazon-qa.s3-us-west-2.amazonaws.com"`
  - `class DownloadError(Exception)`
  - `ensure_file(path: Path, url: str, sha256: str) -> str`：返回 `"present"`（已存在且 SHA 一致，不联网）或 `"downloaded"`；先写到 `path` 加 `.part` 后缀的文件，校验通过再改名；SHA 不符、HTTP 错误、已有文件 SHA 不符都抛 `DownloadError`（消息里有路径或 URL），并且不留下 `.part`、不动已有文件。进度写到 stderr。
  - `main(argv: Sequence[str] | None = None) -> int`：`--manifest`（必填）、`--base-url`（默认 `AMAZONQA_BASE_URL`）；对清单 `source` 里的两份文件，URL 取 `{base_url}/{文件名}`，路径按当前目录解析（和导入命令一致），缺目录就创建；成功返回 0，`DownloadError` 打印中文说明到 stderr 并返回 2。
  - `python -m reviewproof.ingest.download --manifest samples/manifest_v1.json`

- [ ] **Step 1: 写失败的测试。** 测试在线程里用 `http.server.ThreadingHTTPServer`（端口 0）服务一个临时目录：
  - `test_downloads_and_verifies`：服务器上有文件 → 返回 `"downloaded"`，内容一致，没有 `.part`。
  - `test_present_file_with_matching_sha_is_not_fetched`：本地已有正确文件、服务器上没有（会 404）→ 返回 `"present"`。
  - `test_sha_mismatch_after_download_leaves_no_file`：服务器内容和 SHA 不符 → `DownloadError`，目标文件和 `.part` 都不存在。
  - `test_existing_file_with_wrong_sha_is_refused_and_kept`：本地文件内容不对 → `DownloadError`，文件内容不变。
  - `test_http_error_names_the_url`：404 → `DownloadError`，消息含 URL。
  - `test_main_fetches_both_manifest_files`：用 `make_manifest()` 改写 `source` 为临时目录下的绝对路径和真实 SHA，`main(["--manifest", ..., "--base-url", 本地服务器])` 返回 0，两份文件都在；再改一个 SHA 后返回 2。
- [ ] **Step 2:** `.venv/bin/python -m unittest tests.test_download -b`。Expected：ImportError（模块不存在）。
- [ ] **Step 3:** 按 Interfaces 实现 `reviewproof/ingest/download.py`（`urllib.request` 流式读取，1 MiB 一块）。
- [ ] **Step 4:** 同一命令。Expected：6 个测试通过。
- [ ] **Step 5:** 不下载、只确认真实地址可用：`curl -sI` 两个 URL，Expected：`HTTP/1.1 200` 且 `Content-Length` 等于 Global Constraints 里的字节数。在本仓库运行 `python -m reviewproof.ingest.download --manifest samples/manifest_v1.json`，Expected：两份都是已存在，没有联网下载，退出码 0。
- [ ] **Step 6: 提交** `feat: download the AmazonQA source files with SHA-256 checks`

### Task 2: 镜像和全栈 Compose，全新环境验证

**Files:**
- Create: `Dockerfile`、`.dockerignore`
- Modify: `docker-compose.yml`
- Test: `tests/test_docker_config.py`

**Interfaces:**
- Consumes: Task 1 的 `python -m reviewproof.ingest.download`；现有 `python -m reviewproof.ingest`、`python -m reviewproof.api --host --port`。
- Produces（README 和技术报告引用这些名字）：
  - 服务 `db`（不变，宿主端口 `127.0.0.1:${REVIEWPROOF_DB_PORT:-54329}`）、`ingest`、`api`（宿主端口 `127.0.0.1:${REVIEWPROOF_API_PORT:-8000}`）。
  - 命名卷 `reviewproof-pgdata`（不变）、`reviewproof-models`（挂到 `/models`，`HF_HOME=/models`）。
  - 一条命令：`docker compose up --build`。

- [ ] **Step 1: 写失败的测试** `tests/test_docker_config.py`（用 `yaml`，它随 transformers 安装）：
  - `test_dockerignore_keeps_secrets_and_data_out`：`.dockerignore` 的条目包含 `.env`、`.env.*`、`data`、`.venv`、`frontend/node_modules`。
  - `test_api_waits_for_ingest_and_db`：`api.depends_on` 里 `ingest` 是 `service_completed_successfully`，`db` 是 `service_healthy`；`ingest.depends_on.db` 是 `service_healthy`。
  - `test_host_ports_bind_to_localhost`：`db` 和 `api` 的每个端口映射都以 `127.0.0.1:` 开头。
  - `test_api_key_comes_from_optional_env_file`：`api` 的 `env_file` 是 `[{"path": ".env", "required": False}]`；整个 compose 文件里没有 `sk-ant`。
- [ ] **Step 2:** `.venv/bin/python -m unittest tests.test_docker_config -b`。Expected：失败（缺 `.dockerignore`、缺服务）。
- [ ] **Step 3: 写 `Dockerfile`。** 两阶段：`node:24-slim` 里 `npm ci && npm run build`；`python:3.12-slim` 里先 `pip install torch --index-url https://download.pytorch.org/whl/cpu`（和 CI 一样），再装 `requirements.txt`，复制 `reviewproof/`、`samples/` 和构建好的 `frontend/dist`；`WORKDIR /app`，`ENV PYTHONUNBUFFERED=1 HF_HOME=/models`。镜像里不放数据、不放 `.env`。
- [ ] **Step 4: 写 `.dockerignore`、改 `docker-compose.yml`。** `ingest`：同一镜像，`./data:/app/data`、`reviewproof-models:/models`，`REVIEWPROOF_DATABASE_URL=postgresql://reviewproof:reviewproof@db:5432/reviewproof`，命令先下载再导入（`sh -c "... download ... && ... ingest ..."`）。`api`：同一镜像，命令 `python -m reviewproof.api --host 0.0.0.0 --port 8000`，同样的数据库地址和模型卷，`env_file` 可选的 `.env`，健康检查用 Python `urllib` 请求 `/healthz`（`start_period` 300 秒，因为要加载模型）。
- [ ] **Step 5:** Step 2 的命令和 `docker compose config -q`。Expected：4 个测试通过，配置无报错。
- [ ] **Step 6: 构建镜像。** `docker compose build`（后台运行，日志放 scratchpad）。Expected：成功；`docker run --rm --entrypoint ls <镜像> -a /app` 里没有 `.env`、`data`。
- [ ] **Step 7: 全新环境验证（不调用 Claude、不下载模型）。**
  1. `git clone` 当前分支到 scratchpad 里的空目录（没有 `.env`）；用 `cp -c` 把两份数据文件克隆进它的 `data/`（APFS 克隆，不占空间），这样下载步骤会校验后跳过。
  2. 在 scratchpad 写一个**不提交**的 override 文件，把 `reviewproof-models` 换成绑定本机 `~/.cache/huggingface`。
  3. `docker compose -p reviewproof-fresh -f docker-compose.yml -f <override> up --build -d`，设 `REVIEWPROOF_DB_PORT=54339`、`REVIEWPROOF_API_PORT=8001`，不碰开发用的数据库卷。
  4. 等 `ingest` 结束（后台，预计 40–60 分钟）。Expected：退出码 0，导入报告里 100 件商品、每个块都有向量。
  5. Expected：`curl 127.0.0.1:8001/healthz` 显示 `database: true`、`anthropic_key_set: false`；`/v1/products` 返回 100 件；在 `api` 容器里 `python -m reviewproof.retrieval --qid 17020` 有候选；`POST /v1/qa` 以 `done` 结束，答案是"服务出错"那句固定文案（Review Focus 1）；用浏览器打开 `http://127.0.0.1:8001/` 截图。
  6. `docker stats --no-stream` 记下 `api` 的内存（Review Focus 5）；记下导入耗时、镜像大小。
  7. 清理：`docker compose -p reviewproof-fresh down -v`，删除克隆目录。
- [ ] **Step 8: 提交** `feat: run the full stack with one Docker Compose command`

### Task 3: README 重写

**Files:**
- Modify: `README.md`；本地 `CLAUDE.md`（不跟踪）的 Commands 一节

**Interfaces:**
- Consumes: Task 2 的服务名、端口变量、命令；Task 2 Step 7 实测的耗时、内存、镜像大小。

- [ ] **Step 1: 写 README。** 顺序：一句话说明 + 截图；**Docker 快速开始**（前提：Docker、可用内存按实测写、磁盘约多少；`cp .env.example .env` 填 key 可选；`docker compose up --build`；第一次要下载约 1.5 GB 数据和模型、导入约多少分钟；之后再启动多久；没有 key 时能浏览、提问会显示服务出错）；本地开发（保留现有命令）；当前进度和金标结果（保留现有内容）；评测、测试；文档索引（定位、路线、复盘、技术报告、标注规范）；**数据来源**（AmazonQA 论文引用、官方 S3 地址、未声明许可所以本仓库不分发数据、评论 ID 是系统生成的）。若仓库里没有 `.env.example` 就一起加上（只有变量名，没有值）。
- [ ] **Step 2:** 逐条核对 README 里的命令能运行、数字有出处：`grep -n "0\.[0-9]" README.md` 每个数都能在报告里找到。
- [ ] **Step 3:** 本地 `CLAUDE.md` 的 Commands 加上 `docker compose up --build` 和下载命令（文件不跟踪，同步备份到 `.git/CLAUDE.md.local-backup`）。
- [ ] **Step 4: 提交** `docs: rewrite the README around the one-command Docker start`

### Task 4: 技术报告

**Files:**
- Create: `docs/technical_report_cn.md`
- Modify: `README.md`（文档索引加链接）、`docs/design_and_lessons_cn.md`（开头指向技术报告）

- [ ] **Step 1: 写报告。** 面向第一次看的读者，结构：摘要；问题与定位（引用定位文档）；数据与样本（100 件商品的抽样、切分、全量上下文和混合检索的分界）；系统（导入、检索、工作流、接口，各一段，附流程图）；评测方法（银标为什么不能下结论、金标怎么做、指标、基线 B0/B1/B2、bootstrap 区间）；结果（检索和可答性的金标表，照抄 `metrics.json`）；错误分析（误拒、是非题结论、9.5 节没有成立的修复）；局限和有效性威胁（一位人工、锚定、dev 用于调参、test 看过错例、题量）；复现（命令、成本、耗时）；后续（第 6 阶段）。设计取舍和问题细节不重复，链接复盘文档。
- [ ] **Step 2:** 用脚本把报告里的每个数字和 `reports/eval/gold-dev`、`gold-test`、`gold-dev-r2`、`gold-dev-r3` 的 `metrics.json` 对一遍；找不到出处的数字删掉或改成有出处的。
- [ ] **Step 3: 提交** `docs: add the technical report`

### Task 5: 公开前检查（只读，交给用户决定）

- [ ] **Step 1:** 全部历史里找密钥：`git log -p --all` 搜 `sk-ant`、`ANTHROPIC_API_KEY=` 后面跟值、`BEGIN .*PRIVATE KEY`；确认 `.env`、`data/` 从未提交（`git log --all --oneline -- .env data`）。
- [ ] **Step 2:** 列出仓库里含 AmazonQA 原文或转述的路径和大致数量（`reports/eval/*/per_question.jsonl` 的答案、`annotations/gold_v1` 的理由和候选池、文档里的例题）。
- [ ] **Step 3:** 其他公开影响：提交里的邮箱、文档里的本机路径和私有仓库链接（`xuyul6/BRIDGE`）、有没有 LICENSE、`.claude/` 下的文件。
- [ ] **Step 4:** 把发现和可选做法（直接公开 / 先加 LICENSE 和数据说明 / 保持私有）报告给用户；不做任何改动。
