# 第 4 阶段：接口和界面 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 做出可以演示的完整系统：在网页上选商品、提问，边等边看进度；拿到带引用的三态回答，点开引用能看原评论，还能查看这次运行的详情。每次运行都存进 PostgreSQL，可以按 `run_id` 回查。

**Architecture:** 第 3 阶段的工作流新增一个流式入口，每完成一个节点产出一条 trace。FastAPI 服务启动时只加载一次向量模型和重排模型，提供以下接口：
- `GET /v1/products`
- `GET /v1/products/{id}/questions`
- `POST /v1/qa`（SSE：先推进度事件，答案核验后再分块发布，不是逐 token 生成）
- `GET /v1/runs/{run_id}`
- `GET /v1/evidence/{review_id}`
- `GET /healthz`

每次运行写入新表 `qa_runs`。Streamlit 页面只调用这些接口，不直接连数据库。另提供一个 MCP 服务，向外部 LLM 客户端暴露商品列表、评论检索和读取评论原文三个只读工具。

**Tech Stack:** FastAPI 0.162、uvicorn、Streamlit 1.66（`streamlit.testing.v1.AppTest` 做页面测试）、httpx（接口客户端和测试）、mcp 2.x（`mcp.server.mcpserver.MCPServer`；2.x 把 `FastMCP` 改名为 `MCPServer`）、LangGraph 1.2 的 `stream(stream_mode="updates")`、unittest。

**Spec:** [docs/product_positioning_cn.md](../../product_positioning_cn.md)（第 3 节用户流程、F1 商品目录、F5 引用可点开、F10 流式输出与进度、F11 运行记录回查）和 [docs/roadmap_cn.md](../../roadmap_cn.md)（第 2 节"流式""运行记录"两项决定、第 3 节阶段 4、第 4 节可参考的旧代码）。

## Global Constraints

- 在分支 `phase-4-api-ui` 上工作，每个任务一次提交。
- `requirements.txt` 新增 `fastapi>=0.162,<1`、`uvicorn[standard]>=0.54,<1`、`streamlit>=1.66,<2`、`httpx>=0.28,<1`、`mcp>=2.3,<3`。
- SSE 事件名固定为 `run_started / progress / answer_chunk / citations / done / error`，格式为 `event: <name>\ndata: <JSON>\n\n`。答案在工作流结束、引用核验完成后才分块发布。文档和界面都要说明"进度实时推送，答案核验后分段显示，不是逐字生成"。
- 服务故障（`service_status = failed`）照常走完事件流，在 `done` 里报告，HTTP 状态码仍是 200。只有调用方的错误才返回 4xx：商品不存在返回 404，问题为空返回 422。`error` 事件只用于工作流本身抛出的意外异常。
- 向量模型和重排模型在服务启动时加载一次（预热），之后每个请求复用。原型阶段用一把进程级锁串行执行工作流：`BM25Cache` 不是线程安全的，CPU 推理也不适合并发。这个限制要写进 CLAUDE.md。
- 每次 `POST /v1/qa` 都写一行 `qa_runs`，包括服务故障的运行。写库失败不能影响已经发给用户的答案：记一条日志，`done` 里的 `persisted` 为 `false`。
- 防泄漏规则不变：`GET /v1/products/{id}/questions` 只返回 `qid` 和 `question_text`，任何接口、页面或 MCP 工具都不返回社区回答、片段和可答性标签。
- 商品没有名称（数据里没有），界面显示 ASIN、品类和评论数，并标注数据来源"AmazonQA 验证集抽样"。系统评论 ID 不能称为 Amazon 评论 ID。
- 页面只通过接口取数据，不导入 `reviewproof.db` 等后端模块。接口地址由 `REVIEWPROOF_API_URL` 配置，默认 `http://127.0.0.1:8000`。
- 测试不调用真实 Claude，也不加载真实模型：用 `FakeLLM`、`FakeEmbedder`、`FakeReranker`，接口测试用 FastAPI 的 `TestClient`，页面测试用 `AppTest`。
- 可以参考旧仓库里的 `app/api/main.py`、`app/streamlit_app.py` 和 `bridge/mcp_server.py`（路线图第 4 节），但不能复制。

## Review Focus

1. **客户端在流式回答中途断开：** 工作流仍然跑完，`qa_runs` 仍然写入；服务端不能因为写给已断开的连接而报错。（任务 3 测试）
2. **工作流内部服务故障**（例如 Claude 不可用）：SSE 依然以 `done` 结束，`service_status = failed`、HTTP 200；页面显示"服务出错"，不显示"评论无法确认"。（任务 3、任务 4 测试）
3. **`qa_runs` 写库失败：** 用户照样拿到完整答案，`done.persisted = false`。（任务 3 测试）
4. **两个请求同时到达：** 两次运行都完成，结果互不串扰（各自的 `run_id`、引用和 trace）。（任务 3 测试）
5. **引用的评论在数据库里已不存在**（重新导入后被删）：`GET /v1/evidence/{review_id}` 返回 404，页面提示"原评论已不在当前数据中"，不崩溃。（任务 2、任务 4 测试）

---

### Task 1: 运行记录表与流式工作流入口

**Files:**
- Modify: `reviewproof/schema.sql`、`reviewproof/db.py`（`ALL_TABLES` 最前面加 `qa_runs`）、`reviewproof/workflow/graph.py`、`reviewproof/workflow/__init__.py`
- Create: `reviewproof/runs.py`
- Test: `tests/test_runs.py`、`tests/test_workflow.py`（新增流式入口的测试）

**Interfaces:**
- 新表：

```sql
CREATE TABLE IF NOT EXISTS qa_runs (
  run_id TEXT PRIMARY KEY, product_id TEXT NOT NULL REFERENCES products,
  question TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  route TEXT NOT NULL, status TEXT, service_status TEXT NOT NULL, verdict TEXT,
  llm_model TEXT NOT NULL, latency_ms DOUBLE PRECISION NOT NULL, result JSONB NOT NULL);
CREATE INDEX IF NOT EXISTS qa_runs_product_idx ON qa_runs (product_id, created_at DESC);
```

- Produces（`graph.py`）：`stream_answer(conn, product_id, question, *, llm, retrieval, max_rewrites=1) -> Iterator[TraceEntry | QAResult]`。它的校验和 `answer_question` 相同（问题为空抛 `ValueError`，商品不存在抛 `ProductNotFoundError`），在第一次 `next()` 时执行。之后用 `graph.stream(..., stream_mode="updates")` 运行，每当某个节点的更新里带有 `trace` 项，就逐条产出 `TraceEntry`，最后产出一个 `QAResult`。`answer_question` 改为消费 `stream_answer` 并返回最后的 `QAResult`；已有测试不改，全部照常通过。
- Produces（`runs.py`）：
  - `new_run_id() -> str`：`uuid4().hex`。
  - `save_run(conn, run_id: str, result: QAResult, llm_model: str) -> None`：`result` 列存 `dataclasses.asdict(result)`，提交事务。
  - `get_run(conn, run_id: str) -> dict | None`：返回 `{"run_id", "created_at"（ISO 字符串）, "llm_model", **result}`，并回滚只读事务。

- [ ] **Step 1: 写失败测试**
  - `test_runs.py`（DatabaseTestCase）：
    - `test_save_and_get_round_trip`：保存后取回，`answer_text`、`citations`、`trace` 都一致。
    - `test_get_unknown_returns_none`
    - `test_failed_run_is_saved`：`status=None`、`service_status=failed` 也能保存并取回。
  - `test_workflow.py`：
    - `test_stream_yields_trace_then_result`：用 answered 路径的脚本，产出的元素依次是 5 个 `TraceEntry`（plan、retrieve、assess、generate、verify），最后一个是 `QAResult`，并且结果的 `trace` 与这 5 条相同。
    - `test_stream_validates_on_first_next`：未知商品时，`next()` 抛 `ProductNotFoundError`。
- [ ] **Step 2:** 运行，预期 FAIL。
- [ ] **Step 3: 实现。**
- [ ] **Step 4:** 运行，预期 PASS；全量测试 PASS（包括第 3 阶段原有的全部工作流测试）。
- [ ] **Step 5: 提交** `feat: add qa_runs table and a streaming workflow entry point`

---

### Task 2: 服务组装与只读接口

**Files:**
- Create: `reviewproof/api/__init__.py`、`reviewproof/api/services.py`、`reviewproof/api/queries.py`、`reviewproof/api/app.py`
- Modify: `requirements.txt`、`tests/support.py`（新增 `make_test_services(llm)`）
- Test: `tests/test_api.py`

**Interfaces:**
- Produces（`services.py`）：

```python
@dataclass
class Services:
    settings: Settings
    llm: LLM
    retrieval: RetrievalDeps
    lock: threading.Lock = field(default_factory=threading.Lock)
    def connect(self) -> psycopg.Connection          # 每个请求新开一个连接，用完关闭

def build_services(settings: Settings) -> Services   # 用第 3 阶段命令行的 default_llm / default_embedder / default_reranker
def warm_up(services: Services) -> None              # embed_query("warm up") + reranker.score("warm up", ["warm up"])，让模型在启动时加载
```

- Produces（`app.py`）：`create_app(services: Services | None = None) -> FastAPI`。传入 `services` 时直接使用，不预热（给测试用）。为 `None` 时在 lifespan 里依次调用 `build_services(load_settings())` 和 `warm_up`。只在 `reviewproof/api/__main__.py` 里加载 `.env`。
- Produces（`tests/support.py`）：`make_test_services(llm=None) -> Services`。用测试库，`FakeEmbedder`、`FakeReranker`，`llm` 默认是 `FakeLLM({})`。

- Produces（`queries.py`，接口和任务 5 的 MCP 工具共用）：`list_products(conn, *, full_context_max_tokens: int, category=None, split=None) -> dict`、`list_questions(conn, product_id, limit=20) -> dict | None`、`get_review(conn, review_id) -> dict | None`。返回值就是下表里的 JSON 结构；商品或评论不存在时返回 `None`。

接口（本任务只做只读接口）：

| 方法与路径 | 返回 |
|---|---|
| `GET /healthz` | `{"database": bool, "llm_model": str, "anthropic_key_set": bool}`；数据库连不上时 `database = false`，仍返回 200 |
| `GET /v1/products?category=&split=` | `{"source": "AmazonQA validation sample (<dataset_id>)", "products": [{"product_id", "category", "title": null, "review_count", "chunk_count", "split", "strategy": "full_context" | "hybrid"}]}`，按品类、再按评论数降序排列；`strategy` 按 `token_estimate <= full_context_max_tokens` 判断 |
| `GET /v1/products/{id}/questions?limit=20` | `{"product_id", "questions": [{"qid", "question_text"}]}`，按 `qid` 排序；商品不存在返回 404 |
| `GET /v1/evidence/{review_id}` | `{"review_id", "product_id", "text", "helpful_yes", "helpful_total", "chunks": [{"chunk_id", "start_offset", "end_offset"}]}`；不存在返回 404，`detail` 为 `"review not found in the current data"` |
| `GET /v1/runs/{run_id}` | `get_run` 的结果；不存在返回 404 |

- [ ] **Step 1: 写失败测试 `tests/test_api.py`**（DatabaseTestCase + `TestClient(create_app(make_test_services()))`）：
  - `test_healthz`
  - `test_products_list_and_filters`：两件商品，`category` 和 `split` 过滤都生效，`strategy` 正确。
  - `test_questions_only_text`：响应 JSON 里不含 `SECRET-ANSWER`、`SECRET-SNIPPET`、`is_answerable`。
  - `test_questions_unknown_product_404`
  - `test_evidence_returns_review_and_chunks`
  - `test_evidence_missing_404`
  - `test_run_round_trip_and_404`：先用 `save_run` 存一条再取回。
- [ ] **Step 2:** 运行，预期 FAIL。
- [ ] **Step 3: 实现，并安装新依赖。**
- [ ] **Step 4:** 运行，预期 PASS；全量测试 PASS。
- [ ] **Step 5: 提交** `feat: add API app with products, questions, evidence and run lookup`

---

### Task 3: 流式问答接口 `POST /v1/qa`

**Files:**
- Create: `reviewproof/api/qa.py`（事件生成）
- Modify: `reviewproof/api/app.py`
- Test: `tests/test_api_qa.py`

**Interfaces:**
- 请求体：`{"product_id": str, "question": str}`（Pydantic 模型；`question` 去掉首尾空白后为空时返回 422）。商品不存在时，在开始流式输出前返回 404。
- Produces：`qa_events(services: Services, product_id: str, question: str) -> Iterator[tuple[str, dict]]`，产出 `(事件名, 数据)`。它是同步生成器，路由用 `StreamingResponse(media_type="text/event-stream")` 包装，每个事件格式化为 `event: {name}\ndata: {json}\n\n`（`ensure_ascii=False`）。

事件顺序与内容：
1. `run_started`：`{"run_id", "product_id", "question"}`。
2. 每个 `TraceEntry` 产出一次 `progress`：`{"node", "status", "detail", "latency_ms"}`。
3. 拿到 `QAResult` 后：
   - `answer_chunk`：把 `answer_text` 按行切开，过长的行再按约 120 个字符切块，每块一个事件 `{"text"}`；行尾的换行保留在块里，客户端把各块直接拼起来就是完整文本。
   - `citations`：`{"citations": [...]}`，即 `asdict` 后的引用列表，可以为空。
   - `done`：`{"run_id", "status", "service_status", "verdict", "route", "unknowns", "search_queries", "retrieval_modes", "rewrites_used", "llm_calls", "input_tokens", "output_tokens", "latency_ms", "persisted"}`。
4. 工作流抛出意外异常时：产出 `error` 事件 `{"run_id", "message": "<异常类型>: <消息>"}` 后结束，不写 `qa_runs`。

规则：
- 工作流在一个 `threading.Thread` 里运行。线程从工作流开始到 `save_run` 结束都持有 `services.lock`，并自己打开、关闭数据库连接。它通过 `queue.Queue` 把 `TraceEntry` 逐条交给生成器；最后一项是 `(QAResult, persisted)`，或者意外异常对象。
- 生成器在启动线程之前先产出 `run_started`，之后从队列取数据：收到 `TraceEntry` 就产出 `progress`，收到最终结果就依次产出 `answer_chunk`、`citations`、`done`。
- 线程先写 `qa_runs` 再交出最终结果，所以发出 `answer_chunk` 时已经写入；写库失败时记日志，`persisted = false`，流程照常继续。
- 客户端断开时，生成器被关闭，但线程照常跑完并写库。生成器不 join 线程，也不向已断开的连接写数据。

- [ ] **Step 1: 写失败测试 `tests/test_api_qa.py`**（DatabaseTestCase + `TestClient`；`FakeLLM` 用 answered 路径的脚本）：

```python
def parse_sse(text):   # 测试辅助：返回 [(event, data_dict)]
    ...

def test_event_order_and_answer(self):
    events = self.post_qa("A", "Is it loud?")
    names = [e for e, _ in events]
    self.assertEqual(names[0], "run_started")
    self.assertEqual(names[1:6], ["progress"] * 5)
    self.assertIn("answer_chunk", names)
    self.assertEqual(names[-2:], ["citations", "done"])
    text = "".join(d["text"] for e, d in events if e == "answer_chunk")
    self.assertIn("[1]", text)
    done = events[-1][1]
    self.assertEqual((done["status"], done["service_status"], done["persisted"]), ("answered", "ok", True))
    run = self.client.get(f"/v1/runs/{done['run_id']}").json()
    self.assertEqual(run["answer_text"], text)

def test_service_failure_is_200_and_done(self):        # EvidenceAssessment 脚本为 LLMError → done.service_status == "failed"，文本含 "service error"，qa_runs 有一行
def test_unknown_product_404_before_stream(self): ...
def test_blank_question_422(self): ...
def test_persist_failure_still_answers(self):          # patch reviewproof.api.qa.save_run 抛 psycopg.OperationalError → done.persisted is False，答案完整
def test_unexpected_exception_emits_error_event(self): # patch stream_answer 在第二条 trace 后抛 RuntimeError → 最后一个事件是 error
def test_client_disconnect_still_saves_run(self):
    # 直接调用 qa_events：取到 run_started 和第一条 progress 后 gen.close()；这时线程已持有锁，
    # 测试 `with services.lock:` 即可等到它跑完并写库，然后断言 qa_runs 里有这个 run_id
def test_concurrent_requests_do_not_mix(self):
    # 两个线程各自用 TestClient 提问（商品 A 和 B，各自的 FakeLLM 脚本分别引用自己商品的块）
    # 两个 run_id 不同，各自的引用只属于自己的商品
```

（并发测试里，两个请求共用同一个 `Services`，所以 `FakeLLM` 要能在一份脚本里同时满足两次运行：用按提示词内容决定输出的可调用对象，提示词含 `[A-` 就引用 A 的块，含 `[B-` 就引用 B 的块。）

- [ ] **Step 2:** 运行，预期 FAIL。
- [ ] **Step 3: 实现 `qa.py` 并接上路由。**
- [ ] **Step 4:** 运行，预期 PASS；全量测试 PASS。
- [ ] **Step 5: 提交** `feat: add streaming QA endpoint with run persistence`

---

### Task 4: Streamlit 页面

**Files:**
- Create: `reviewproof/ui/__init__.py`、`reviewproof/ui/client.py`、`reviewproof/ui/app.py`
- Test: `tests/test_ui_client.py`、`tests/test_ui_app.py`

**Interfaces:**
- Produces（`client.py`，只依赖 httpx，不导入后端模块）：
  - `parse_sse(lines: Iterable[str]) -> Iterator[tuple[str, dict]]`：按空行切分事件，支持一个事件里有多行 `data:`。
  - `class ApiClient(base_url: str, transport: httpx.BaseTransport | None = None)`，方法：`health()`、`products(category=None)`、`questions(product_id)`、`evidence(review_id) -> dict | None`（404 时返回 `None`）、`run(run_id)`、`ask(product_id, question) -> Iterator[tuple[str, dict]]`（用 `httpx.Client.stream` 逐行读取）。
  - `api_client() -> ApiClient`：从 `REVIEWPROOF_API_URL` 读接口地址，默认 `http://127.0.0.1:8000`。页面通过这个函数取客户端，测试 patch 它。

页面（`reviewproof/ui/app.py`，运行命令 `streamlit run reviewproof/ui/app.py`）：
- 侧边栏：品类选择框，商品选择框（显示 `ASIN · 评论数 · 全量上下文/混合检索`），数据来源说明，接口健康状态（数据库是否连上、是否配置了 Claude key）。
- 主区：从示例问题里选（来自 `questions` 接口），或者自己输入；点"提问"后，用 `st.status` 实时显示进度（每个节点一行）。
- 结果区：
  - 状态徽标：answered 显示"能回答"，partial 显示"部分能回答"，unanswerable 显示"评论无法确认"，`service_status = failed` 显示"服务出错（不代表评论里没有答案）"，`route = out_of_scope` 显示"不在服务范围"。
  - 答案正文。
  - 每条引用一个折叠块：标题 `[n] chunk_id`；展开后调用 `evidence` 显示原评论全文；返回 `None` 时显示"原评论已不在当前数据中"。
  - "运行详情"折叠块：`run_id`、检索词、检索模式、改写次数、trace 表、token 用量。
- 页脚说明：进度实时显示，答案核验完成后分段显示，不是逐字生成；评论 ID 是系统生成的，不是 Amazon 评论 ID。
- 接口不可达时显示错误提示和启动命令，不崩溃。

- [ ] **Step 1: 写失败测试**
  - `test_ui_client.py`（离线，用 `httpx.MockTransport`）：
    - `test_parse_sse_multiline_and_blank_separation`
    - `test_ask_streams_events`
    - `test_evidence_404_returns_none`
    - `test_products_passes_category`
  - `test_ui_app.py`（`AppTest.from_file("reviewproof/ui/app.py")`，并 patch `reviewproof.ui.client.api_client` 返回一个假客户端）：
    - `test_renders_products`：页面上有商品选择框，选项里有 ASIN。
    - `test_ask_shows_answer_badge_and_citation`：点击"提问"后，页面出现"能回答"和 `[1]`。
    - `test_service_failure_badge`：假客户端的 `done.service_status = failed` 时，页面显示"服务出错"，不显示"评论无法确认"。
    - `test_missing_evidence_message`：`evidence` 返回 `None` 时显示"原评论已不在当前数据中"。
    - `test_api_unreachable`：`products` 抛 `httpx.ConnectError` 时显示启动提示，不抛异常。
- [ ] **Step 2:** 运行，预期 FAIL。
- [ ] **Step 3: 实现。** 引用的原评论在展开时才请求，并用 `st.cache_data` 缓存。
- [ ] **Step 4:** 运行，预期 PASS；全量测试 PASS。
- [ ] **Step 5: 提交** `feat: add Streamlit UI on the QA API`

---

### Task 5: MCP 检索服务

**Files:**
- Create: `reviewproof/mcp_server.py`
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- Produces：`build_server(services: Services) -> MCPServer`（`from mcp.server.mcpserver import MCPServer`，名称 `reviewproof`）。注册三个只读工具：
  - `list_products(category: str | None = None) -> list[dict]`：与 `GET /v1/products` 的商品项相同。
  - `search_reviews(product_id: str, query: str) -> dict`：调用第 2 阶段的 `retrieve`（`strategy="auto"`），返回 `{"strategy", "mode", "service_status", "results": [{"chunk_id", "review_id", "text", "dense_rank", "bm25_rank", "rerank_score"}]}`。商品不存在时返回 `{"error": "product not found"}`。
  - `get_review(review_id: str) -> dict`：与 `GET /v1/evidence/{review_id}` 相同；不存在时返回 `{"error": "review not found in the current data"}`。
- 工具复用任务 2 的 `reviewproof/api/queries.py`，与接口返回同样的结构。
- `main()`：用 `build_services(load_settings())` 和 `warm_up` 组装服务，`server.run(transport=os.environ.get("REVIEWPROOF_MCP_TRANSPORT", "stdio"))`。命令是 `python -m reviewproof.mcp_server`，`.env` 只在 `if __name__ == "__main__"` 里加载。
- 不提供问答工具：调用 MCP 的客户端自己就是 LLM，只需要检索到的证据。

- [ ] **Step 1: 写失败测试 `tests/test_mcp_server.py`**（DatabaseTestCase；`make_test_services()`）：
  - `test_tools_registered`：`asyncio.run(server.list_tools())` 返回的工具名正好是这三个。
  - `test_search_reviews_scoped_to_product`：结果都属于该商品，第一条是 `A-r00001-c00`。
  - `test_unknown_product_and_review_return_error`
  - `test_no_community_data_in_any_tool_output`：三个工具的输出里都不含 `SECRET`。

  直接调用模块里被装饰的函数即可（`MCPServer.tool` 返回原函数）。
- [ ] **Step 2:** 运行，预期 FAIL。
- [ ] **Step 3: 实现。**
- [ ] **Step 4:** 运行，预期 PASS；全量测试 PASS。
- [ ] **Step 5: 提交** `feat: add read-only MCP retrieval server`

---

### Task 6: 端到端运行与文档

**Files:**
- Create: `reviewproof/api/__main__.py`（`load_dotenv()` 后启动 `uvicorn.run(create_app(), host=..., port=...)`，参数 `--host 127.0.0.1 --port 8000`），`.claude/launch.json`（`api` 和 `ui` 两项，供预览）
- Modify: `CLAUDE.md`、`README.md`、`.env.example`（`REVIEWPROOF_API_URL`、`REVIEWPROOF_MCP_TRANSPORT`）

- [ ] **Step 1: 启动。** 先运行 `python -m reviewproof.api`，等 `/healthz` 返回 `database: true`（启动时会预热模型，约 15 秒），再运行 `streamlit run reviewproof/ui/app.py`。
- [ ] **Step 2: 真实端到端验证。** 会调用真实的 Claude，**先征得用户同意**。在浏览器里用 dev 集的两道题（一件 small 商品、qid 17020）走完整流程：
  - 选商品，看到进度逐条出现。
  - 拿到答案和状态徽标。
  - 点开一条引用，看到原评论。
  - 打开运行详情，再用 `GET /v1/runs/{run_id}` 取回同一条记录。

  在台账里记录：第二次提问的耗时（模型已预热，应明显短于第 3 阶段命令行的 30 秒）、事件顺序、界面截图要点。
- [ ] **Step 3: 更新文档。**
  - `CLAUDE.md`：Commands 加 API、UI、MCP 三条命令；Architecture 加"Phase 4"一段：接口表、SSE 事件、`qa_runs`、启动预热、进程级锁（同一时间只跑一个工作流）、页面只走接口。
  - `README.md`：当前进度加第 4 阶段，并说明怎么启动接口和页面。
- [ ] **Step 4:** 全量测试 PASS。
- [ ] **Step 5: 提交** `feat: add API entry point and document phase 4`
