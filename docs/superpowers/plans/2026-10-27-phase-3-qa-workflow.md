# 第 3 阶段：LangGraph 问答工作流 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给定一件商品和一个问题，产出带引用的三态回答（`answered / partial / unanswerable`）。是非题另给结论。服务故障单独标记，不能报成"评论无法确认"。

**Architecture:** 新增 `reviewproof/llm.py`（Claude 结构化输出客户端）和 `reviewproof/workflow/` 包。工作流是一张 LangGraph 状态图，节点顺序如下：
1. 查询规划：一次 LLM 调用，判断是否在服务范围、是否是非题，并写出英文检索词。
2. 检索：调用第 2 阶段的 `retrieve`。
3. 证据评估：一次 LLM 调用，给出三态、支持证据和缺失部分。
4. 必要时改写检索词，重新检索一次。
5. 带引用生成：一次 LLM 调用，输出逐条结论和引用 ID。
6. 引用核验：纯代码。
7. 汇总：纯代码，渲染答案文本。

降级都在节点内处理，不另写一套备用流程。运行记录（`trace`）放进结果对象；写入数据库是第 4 阶段的事。

**Tech Stack:** Python 3.10+、LangGraph 1.x、Anthropic Python SDK 2.x（`messages.parse` + Pydantic 结构化输出）、Pydantic 2、第 2 阶段的 `reviewproof.retrieval`、unittest。

**Spec:** [docs/product_positioning_cn.md](../../product_positioning_cn.md)（F4 可答性、F5 引用、F6 是非题结论、F8 改写重检索、F9 冲突证据、2.3 节和第 7 节不服务的问题、5.5 节防泄漏规则）和 [docs/roadmap_cn.md](../../roadmap_cn.md)（第 1 节在线流程、第 2 节"生成""可答性""编排"三项决定、第 3 节阶段 3）。

## Global Constraints

- 在分支 `phase-3-workflow` 上工作，每个任务一次提交。
- 生成和评估统一用 Claude Haiku 5.5，模型 ID 是 `claude-haiku-5-5`（不要加日期后缀）。调用走官方 `anthropic` SDK 的 `client.messages.parse(..., output_format=<Pydantic 模型>, output_config={"effort": <effort>})`。不传 `temperature`、`top_p`、`thinking`，也不用 assistant 预填。每次调用 `max_tokens = 4096`（Haiku 5.5 默认开启思考，思考也计入这个上限）。
- 新增配置：`REVIEWPROOF_LLM_MODEL`（默认 `claude-haiku-5-5`）、`REVIEWPROOF_LLM_EFFORT`（默认 `low`，只能是 `low / medium / high`）、`REVIEWPROOF_MAX_REWRITES`（默认 `1`，只能是 0–2）。API key 由 SDK 从 `ANTHROPIC_API_KEY` 读取，不存进 `Settings`，也不打印。
- `requirements.txt` 新增 `anthropic>=2.3,<3`、`langgraph>=1.2,<2`、`pydantic>=2.8,<3`。
- 测试不调用真实 API：LLM 用本计划新增的 `tests.support.FakeLLM`；检索沿用 `FakeEmbedder`、`FakeReranker` 和测试库。
- 状态值：可答性 `answered / partial / unanswerable`；服务状态 `ok / degraded / failed`；是非题结论 `yes / no / mixed / unknown`，非是非题为 `None`。服务状态是 `failed` 时，可答性和是非题结论都是 `None`，答案文本必须说明是服务出错，不能说"评论无法确认"。
- **防泄漏：** 送进任何提示词的只有两样东西：用户这次提的问题文本，以及当前商品的块文本和块 ID。`questions` 表里的社区回答、`review_snippets`、可答性标签、`question_type`，以及同一商品的其他问题，一律不能进入提示词。用 `--qid` 取题时，只取 `question_text` 和 `product_id`。
- 引用只能是本次运行候选里的 `chunk_id`，并且属于当前商品。系统评论 ID 和块 ID 不是 Amazon 评论 ID。
- 改写重检索最多 `REVIEWPROOF_MAX_REWRITES` 次。改写后没有新增证据时立即停止。全量上下文策略的商品不做改写（已经给了全部评论）。
- LangGraph 是唯一的编排方式，不写"图失败就走手写流程"的备用路径。图本身编译或运行出错时，异常照常抛给调用方。

## Review Focus

1. **LLM 引用了不存在的、其他商品的，或不在候选里的块 ID**：这些引用要被丢掉；丢完没有引用的结论也要删掉；`answered / partial` 删到一条结论都不剩时，降为 `unanswerable`，并在 trace 里说明。（任务 2、任务 3 测试）
2. **任何一步出服务故障**（检索 `failed`、证据评估 LLM 失败、生成 LLM 失败、读库异常）：`service_status = failed`，可答性为 `None`，答案文本说明是服务出错。查询规划 LLM 失败则用启发式降级，`service_status = degraded`。（任务 3 测试）
3. **泄漏：** 数据库里同一商品的社区回答、片段和标签，绝不能出现在任何一次 LLM 调用的提示词里。（任务 3 测试）
4. **改写循环的边界：** 评估一直说 `partial`，并一直给出同一个改写词时，最多改写 `max_rewrites` 次；改写词和已用过的检索词相同，或改写后没有新块，就停止。（任务 3 测试）
5. **中文提问：** 检索用规划节点给出的英文检索词；固定文案（未覆盖、服务出错、是非题结论）按问题语言输出中文。（任务 2、任务 3 测试）

---

### Task 1: 配置与 Claude 结构化输出客户端

**Files:**
- Modify: `reviewproof/config.py`、`requirements.txt`、`tests/support.py`（新增 `FakeLLM`）
- Create: `reviewproof/llm.py`
- Test: `tests/test_config.py`、`tests/test_llm.py`

**Interfaces:**
- Produces（`config.py`）：`Settings` 新增 `llm_model: str = "claude-haiku-5-5"`、`llm_effort: str = "low"`、`max_rewrites: int = 1`。取值不合法时抛 `ValueError`，并写出变量名。
- Produces（`llm.py`）：

```python
T = TypeVar("T", bound=BaseModel)

class LLMError(RuntimeError): ...

@dataclass(frozen=True)
class LLMResult(Generic[T]):
    output: T
    input_tokens: int
    output_tokens: int
    latency_ms: float

class LLM(Protocol):
    model_name: str
    def structured(self, *, system: str, prompt: str, schema: type[T]) -> LLMResult[T]: ...

class ClaudeLLM:
    def __init__(self, model: str = "claude-haiku-5-5", effort: str = "low", client=None, timeout: float = 60.0): ...
```

- Produces（`tests/support.py`）：`FakeLLM(script: Mapping[str, Sequence[object]])`。`model_name = "fake-llm"`。按 `schema.__name__` 取脚本里的下一项：如果是 `Exception`，就抛出；如果是可调用对象，就用 `fn(prompt)` 的返回值；否则直接当作输出。每次调用返回 `LLMResult(output, 10, 5, 1.0)`，并把 `(schema.__name__, system, prompt)` 追加到 `self.calls`。脚本用完时抛 `AssertionError`。

`ClaudeLLM` 的规则：
- `client` 为 `None` 时，第一次调用才构造 `anthropic.Anthropic(timeout=timeout)`。
- 构造或请求时出现的任何 `anthropic.AnthropicError`（所有 SDK 异常的基类，包括缺少 key、API 错误、连接错误），都转成 `LLMError(f"{type(exc).__name__}: {exc}")`。
- 调用 `client.messages.parse(model=..., max_tokens=4096, system=system, messages=[{"role": "user", "content": prompt}], output_format=schema, output_config={"effort": effort})`。
- 响应 `stop_reason == "refusal"` 时抛 `LLMError("refused")`；`stop_reason == "max_tokens"` 时抛 `LLMError("truncated")`；`parsed_output` 为 `None` 时抛 `LLMError("no structured output")`。
- token 数取自 `response.usage.input_tokens` 和 `response.usage.output_tokens`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_config.py
def test_llm_defaults_and_validation(self):
    s = load_settings({})
    self.assertEqual((s.llm_model, s.llm_effort, s.max_rewrites), ("claude-haiku-5-5", "low", 1))
    with self.assertRaisesRegex(ValueError, "REVIEWPROOF_LLM_EFFORT"):
        load_settings({"REVIEWPROOF_LLM_EFFORT": "max"})
    with self.assertRaisesRegex(ValueError, "REVIEWPROOF_MAX_REWRITES"):
        load_settings({"REVIEWPROOF_MAX_REWRITES": "3"})

# tests/test_llm.py（离线；FakeAnthropic 记录 parse 的参数，返回 SimpleNamespace 响应）
class Answer(BaseModel):
    text: str

def test_parse_call_shape(self):
    fake = FakeAnthropic(response(parsed=Answer(text="hi"), stop_reason="end_turn", usage=(120, 30)))
    result = ClaudeLLM(client=fake).structured(system="sys", prompt="q", schema=Answer)
    self.assertEqual((result.output.text, result.input_tokens, result.output_tokens), ("hi", 120, 30))
    kwargs = fake.calls[0]
    self.assertEqual(kwargs["model"], "claude-haiku-5-5")
    self.assertEqual(kwargs["output_format"], Answer)
    self.assertEqual(kwargs["output_config"], {"effort": "low"})
    self.assertEqual(kwargs["max_tokens"], 4096)
    self.assertEqual(kwargs["messages"], [{"role": "user", "content": "q"}])
    for banned in ("temperature", "top_p", "thinking"):
        self.assertNotIn(banned, kwargs)

def test_refusal_truncation_and_missing_output_raise(self): ...   # 三种情况各抛 LLMError，消息分别含 "refused" / "truncated" / "no structured output"

def test_sdk_errors_become_llm_error(self):
    error = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    with self.assertRaisesRegex(LLMError, "APIConnectionError"):
        ClaudeLLM(client=FakeAnthropic(error=error)).structured(system="s", prompt="p", schema=Answer)

def test_fake_llm_script(self):
    llm = FakeLLM({"Answer": [Answer(text="a"), LLMError("boom")]})
    self.assertEqual(llm.structured(system="s", prompt="p", schema=Answer).output.text, "a")
    with self.assertRaises(LLMError):
        llm.structured(system="s", prompt="p", schema=Answer)
    self.assertEqual(llm.calls[0][0], "Answer")
```

（`httpx2` 是 anthropic 2.x 自带的依赖；若导入失败，改用 `anthropic.APIConnectionError.__init__` 签名中要求的请求类型。）

- [ ] **Step 2:** 运行 `.venv/bin/python -m unittest tests.test_config tests.test_llm -b`，预期 FAIL。
- [ ] **Step 3: 实现。** 在 `requirements.txt` 新增三个依赖，并在 `.venv` 安装。
- [ ] **Step 4:** 运行同一命令，预期 PASS；全量测试 PASS。
- [ ] **Step 5: 提交** `feat: add Claude structured-output client and LLM settings`

---

### Task 2: 输出模型、提示词、引用核验与答案渲染

**Files:**
- Create: `reviewproof/workflow/__init__.py`、`reviewproof/workflow/schemas.py`、`reviewproof/workflow/prompts.py`、`reviewproof/workflow/answer.py`
- Test: `tests/test_workflow_answer.py`

**Interfaces:**
- Consumes: `Candidate`（第 2 阶段 `reviewproof.retrieval.types`）。
- Produces（`schemas.py`，Pydantic 模型，即 LLM 输出格式）：

```python
class QueryPlan(BaseModel):
    route: Literal["review_question", "out_of_scope"]
    out_of_scope_reason: str          # route 为 review_question 时为空串
    is_yes_no: bool
    search_query: str                 # 英文检索词；中文问题在这里翻成英文

class EvidenceAssessment(BaseModel):
    status: Literal["answered", "partial", "unanswerable"]
    supporting_chunk_ids: list[str]
    missing: list[str]                # 问题中评论没覆盖的部分，用问题的语言写
    rewrite_query: str                # 针对缺失部分的英文检索词；没有就为空串

class Claim(BaseModel):
    text: str                         # 用问题的语言写的一条结论
    chunk_ids: list[str]

class GeneratedAnswer(BaseModel):
    claims: list[Claim]
    unknowns: list[str]
    verdict: Literal["yes", "no", "mixed", "unknown"] | None   # 只在是非题时给出
```

- Produces（`answer.py`，均为纯函数和 frozen dataclass）：

```python
@dataclass(frozen=True)
class Citation:
    number: int                       # 从 1 开始，按首次出现的顺序编号
    chunk_id: str
    review_id: str
    text: str

@dataclass(frozen=True)
class AnswerClaim:
    text: str
    citation_numbers: tuple[int, ...]

def is_chinese(text: str) -> bool     # 含有 CJK 统一汉字（U+4E00–U+9FFF）即为真

def verify_claims(claims: Sequence[Claim], candidates: Sequence[Candidate], product_id: str
                  ) -> tuple[tuple[AnswerClaim, ...], tuple[Citation, ...], tuple[str, ...]]
    # 返回 (保留的结论, 引用列表, 被丢掉的 chunk_id)

def final_status(assessed: str, claims: Sequence[AnswerClaim], unknowns: Sequence[str]) -> str

def render_answer(*, question: str, route: str, out_of_scope_reason: str, service_status: str,
                  status: str | None, verdict: str | None, claims: Sequence[AnswerClaim],
                  unknowns: Sequence[str]) -> str
```

规则：
- `verify_claims`：某个 chunk_id 不在 `candidates` 里，或候选的 `product_id` 不等于 `product_id`，就丢掉。去掉无效 ID 后没有引用的结论整条删除。同一个块在多条结论里出现时只编一个号。
- `final_status`：评估为 `unanswerable`，结果就是 `unanswerable`；评估为 `answered / partial` 但核验后没有结论，降为 `unanswerable`；评估为 `answered` 但生成给出了 `unknowns`，降为 `partial`；其余照评估结果。
- `render_answer` 的固定文案（问题含中文用中文，否则用英文）：

| 情况 | 中文 | English |
|---|---|---|
| 服务故障（最优先） | `服务出错，这次没有完成检索或评估，不代表评论里没有答案。` | `A service error stopped this run; this is not a finding about the reviews.` |
| 不在服务范围 | `这个问题不在评论能回答的范围内：{reason}` | `This question is outside what reviews can answer: {reason}` |
| 不可答 | `评论无法确认这个问题。` | `The reviews don't confirm this.` |
| 结论行 | `{text} [1][2]` | 同左 |
| 未覆盖 | `评论没有提到：{unknown1}；{unknown2}` | `Not covered by the reviews: {u1}; {u2}` |
| 是非题结论 | `结论：是 / 否 / 评论意见不一 / 无法确认` | `Verdict: Yes / No / Reviews disagree / Can't confirm` |

  答案文本按下面的顺序拼接，各部分之间用换行分隔：是非题结论（`verdict` 不为 `None` 时）→ 每条结论一行 → 未覆盖（`unknowns` 不为空时）。不可答时只输出是非题结论（如有）和"不可答"那一句。
- `prompts.py` 提供 `plan_prompt(question)`、`assess_prompt(question, candidates)`、`generate_prompt(question, is_yes_no, status, candidates, missing)`，以及三段 system 文本。每段 system 文本必须写清楚的要求：
  - **规划：** 价格、库存、物流、促销、售后政策、官方参数、跨商品对比或推荐、医疗或安全类确定性判断，都判为 `out_of_scope`；`search_query` 一律用英文。
  - **评估：** 只能依据给出的评论块判断；三态的定义要写进去（全部能回答 / 只能回答一部分 / 评论没有相关信息）；不能用常识补全。
  - **生成：** 每条结论必须引用至少一个给出的块 ID；评论之间互相矛盾时，把不同说法分成不同结论并列给出，`verdict` 设为 `mixed`；只在是非题时给 `verdict`；用问题的语言写。
  - 候选格式是每块一段 `[chunk_id] text`。提示词里只出现问题文本和候选块，不出现商品之外的任何数据。

- [ ] **Step 1: 写失败测试 `tests/test_workflow_answer.py`**（离线；用手工构造的 `Candidate(ChunkRow(...))`）

```python
def test_verify_drops_unknown_and_foreign_ids_and_numbers_citations(self):
    claims = [Claim(text="Loud.", chunk_ids=["A-r00001-c00", "B-r00000-c00", "nope"]),
              Claim(text="Only bad ids.", chunk_ids=["nope"]),
              Claim(text="Again loud.", chunk_ids=["A-r00001-c00", "A-r00002-c00"])]
    kept, citations, dropped = verify_claims(claims, CANDS_A_PLUS_FOREIGN_B, "A")
    self.assertEqual([(c.text, c.citation_numbers) for c in kept], [("Loud.", (1,)), ("Again loud.", (1, 2))])
    self.assertEqual([c.chunk_id for c in citations], ["A-r00001-c00", "A-r00002-c00"])
    self.assertEqual(set(dropped), {"B-r00000-c00", "nope"})

def test_final_status_rules(self):
    one = [AnswerClaim("x", (1,))]
    self.assertEqual(final_status("answered", one, []), "answered")
    self.assertEqual(final_status("answered", one, ["size"]), "partial")
    self.assertEqual(final_status("partial", [], []), "unanswerable")
    self.assertEqual(final_status("unanswerable", one, []), "unanswerable")

def test_render_service_failure_never_says_unanswerable(self):
    text = render_answer(question="Is it loud?", route="review_question", out_of_scope_reason="",
                         service_status="failed", status=None, verdict=None, claims=[], unknowns=[])
    self.assertIn("service error", text)
    self.assertNotIn("don't confirm", text)

def test_render_chinese_partial_with_verdict(self):
    text = render_answer(question="声音大吗？", route="review_question", out_of_scope_reason="",
                         service_status="ok", status="partial", verdict="mixed",
                         claims=[AnswerClaim("有人说很吵。", (1,))], unknowns=["夜间使用"])
    self.assertEqual(text, "结论：评论意见不一\n有人说很吵。 [1]\n评论没有提到：夜间使用")

def test_render_unanswerable_and_out_of_scope(self): ...   # 英文不可答一句；中文不在服务范围带原因

def test_prompts_contain_only_question_and_chunks(self):
    p = assess_prompt("Is it loud?", CANDS_A)
    self.assertIn("[A-r00001-c00] Too loud for me.", p)
    self.assertIn("Is it loud?", p)
```

- [ ] **Step 2:** 运行 `.venv/bin/python -m unittest tests.test_workflow_answer -b`，预期 FAIL。
- [ ] **Step 3: 实现 `schemas.py`、`prompts.py`、`answer.py`。**
- [ ] **Step 4:** 运行同一命令，预期 PASS；全量测试 PASS。
- [ ] **Step 5: 提交** `feat: add workflow schemas, prompts, citation checks and answer rendering`

---

### Task 3: LangGraph 工作流

**Files:**
- Create: `reviewproof/workflow/state.py`、`reviewproof/workflow/nodes.py`、`reviewproof/workflow/graph.py`
- Modify: `reviewproof/workflow/__init__.py`（导出 `answer_question`、`QAResult`、`RetrievalDeps`）
- Test: `tests/test_workflow.py`

**Interfaces:**
- Consumes: 任务 1 的 `LLM`、`LLMError`；任务 2 的全部接口；第 2 阶段的 `retrieve`、`RetrievalParams`、`ProductNotFoundError`、`BM25Cache`、`Reranker`、`Embedder`。
- Produces：

```python
@dataclass(frozen=True)
class RetrievalDeps:
    embedder: Embedder
    embedding_model: str
    reranker: Reranker | None
    bm25_cache: BM25Cache
    params: RetrievalParams = RetrievalParams()

@dataclass(frozen=True)
class TraceEntry:
    node: str              # plan / retrieve / assess / rewrite / generate / verify
    status: str            # ok / degraded / failed / skipped
    detail: str
    latency_ms: float

@dataclass(frozen=True)
class QAResult:
    product_id: str
    question: str
    route: str                         # review_question / out_of_scope
    status: str | None                 # answered / partial / unanswerable；服务故障时为 None
    service_status: str                # ok / degraded / failed
    verdict: str | None
    claims: tuple[AnswerClaim, ...]
    unknowns: tuple[str, ...]
    citations: tuple[Citation, ...]
    answer_text: str
    search_queries: tuple[str, ...]    # 实际用过的检索词，按顺序
    retrieval_modes: tuple[str, ...]   # 每次检索的 RetrievalResult.mode
    rewrites_used: int
    trace: tuple[TraceEntry, ...]
    llm_calls: int
    input_tokens: int
    output_tokens: int
    latency_ms: float

def answer_question(conn, product_id: str, question: str, *, llm: LLM, retrieval: RetrievalDeps,
                    max_rewrites: int = 1) -> QAResult
```

- `state.py`：LangGraph 用的 `TypedDict`（`total=False`）。`trace` 和 `usage` 用 `Annotated[list, operator.add]` 做累加。

流程（节点名 → 下一步）：
1. `answer_question` 开始时：`question.strip()` 为空抛 `ValueError`；商品不存在抛 `ProductNotFoundError`。这两种是调用方的错误，不进图。
2. **plan**：调用 `QueryPlan`。出现 `LLMError` 时用启发式降级：`route = review_question`、`search_query = question`；`is_yes_no` 的判断规则是英文以 `is / are / does / do / did / can / will / would / should / could / has / have` 开头（不分大小写），或中文含 `吗`、`是否`、`能不能`、`会不会`。trace 记 `degraded`，最终 `service_status` 至少是 `degraded`。`route == out_of_scope` 时直接到 **finalize**。
3. **retrieve**：用 `search_query` 调 `retrieve(strategy="auto")`，候选取 `result.candidates`。`retrieve` 抛异常，或返回的 `service_status == failed` 时，都记为服务故障并到 **finalize**；返回 `degraded` 时继续，最终至少是 `degraded`。
4. **assess**：候选为空时不调用 LLM，直接判 `unanswerable`，`missing = [question]`。否则调用 `EvidenceAssessment`；不在候选里的 `supporting_chunk_ids` 要过滤掉。出现 `LLMError` 时记为服务故障并到 **finalize**（可答性不用启发式猜）。
5. 评估后分支：满足下面全部条件时进入 **rewrite**：状态是 `partial / unanswerable`、`rewrites_used < max_rewrites`、检索策略是 `hybrid`、`rewrite_query` 不为空，且不在已用过的检索词里。否则，状态为 `answered / partial` 时到 **generate**，为 `unanswerable` 时到 **finalize**（不调用生成，省一次调用）。
6. **rewrite**：`rewrites_used += 1`，用 `rewrite_query` 再调一次 `retrieve`。把新块追加到候选末尾，按 `chunk_id` 去重，总数不超过 `2 × params.evidence_k`。没有新块时 trace 记 `no_new_evidence`，按第 5 步的"否则"分支走，不再评估。有新块时回到 **assess**。重检索出服务故障时，保留已有候选，记 `degraded`，同样按"否则"分支走。
7. **generate**：调用 `GeneratedAnswer`，传入问题、`is_yes_no`、评估状态、全部候选和 `missing`。出现 `LLMError` 时记为服务故障并到 **finalize**。
8. **verify**：`verify_claims` + `final_status`。生成时的 `unknowns` 与评估时的 `missing` 合并去重，作为最终 `unknowns`。被丢掉的 ID 写进 trace。
9. **finalize**：确定 `verdict`：服务故障、不在服务范围或不是是非题时为 `None`；是非题最终为 `unanswerable` 时为 `"unknown"`（包括跳过生成的路径）；其余用生成给出的值，生成没给时为 `"unknown"`。最终状态为 `unanswerable` 时 `unknowns = ()`。然后用 `render_answer` 拼接答案文本，汇总 trace、token 数、LLM 调用次数和总耗时，生成 `QAResult`。不在服务范围时 `status = "unanswerable"`。

`plan`、`retrieve`、`assess`、`rewrite`、`generate`、`verify` 每执行一次就向 `trace` 追加一条 `TraceEntry`；没执行的节点不加，`finalize` 也不加。

- [ ] **Step 1: 写失败测试 `tests/test_workflow.py`**（继承 `DatabaseTestCase`）。`setUp` 准备：
  - 商品 A：用 `make_product("A")`。
  - 商品 C：12 条评论，第 0–5 条讲噪音、第 6–11 条讲电池。
  - 向量：调用 `embed_pending_chunks(FakeEmbedder())`。
  - 题目：给 A 写入一道题，社区回答含 `SECRET-ANSWER`，片段含 `SECRET-SNIPPET`。
  - `RetrievalDeps(params=RetrievalParams(evidence_k=2, full_context_max_tokens=0))`，强制走混合检索并取 2 条证据。

```python
def test_answered_path_cites_only_candidates(self):
    llm = FakeLLM({"QueryPlan": [plan("loud")], "EvidenceAssessment": [assess("answered", ["A-r00001-c00"])],
                   "GeneratedAnswer": [GeneratedAnswer(claims=[Claim(text="A buyer says it is too loud.",
                                                                    chunk_ids=["A-r00001-c00", "B-x"])],
                                                       unknowns=[], verdict="yes")]})
    r = answer_question(self.conn, "A", "Is it loud?", llm=llm, retrieval=self.deps)
    self.assertEqual((r.status, r.service_status, r.verdict), ("answered", "ok", "yes"))
    self.assertEqual([c.chunk_id for c in r.citations], ["A-r00001-c00"])
    self.assertEqual([t.node for t in r.trace], ["plan", "retrieve", "assess", "generate", "verify"])
    self.assertEqual(r.llm_calls, 3)

def test_unanswerable_skips_generation(self):           # 没有 GeneratedAnswer 脚本；status unanswerable，llm_calls == 2，是非题 verdict == "unknown"，unknowns == ()

def test_out_of_scope_skips_retrieval(self):            # route out_of_scope：只有 plan 一条 trace，status unanswerable，文本含原因

def test_partial_then_rewrite_finds_new_evidence(self):
    # 商品 C；评估先说 partial（rewrite_query "battery"），改写后说 answered
    # 断言 rewrites_used == 1、search_queries == ("noise", "battery")、trace 含 rewrite、retrieval_modes 有两项

def test_rewrite_stops_without_new_evidence(self):      # 改写词召回的块全已在候选里 → trace "no_new_evidence"，assess 只调用一次

def test_rewrite_bounded_by_max_rewrites(self):         # 评估一直 partial 且给出新改写词，max_rewrites=1 → rewrites_used == 1
def test_same_rewrite_query_is_not_reused(self):        # rewrite_query 等于原检索词 → 不改写

def test_full_context_product_never_rewrites(self):     # deps 用默认阈值（A 走全量上下文），评估 partial 并给出改写词 → rewrites_used == 0

def test_plan_failure_degrades_with_heuristic(self):    # QueryPlan 脚本为 LLMError → service_status degraded，is_yes_no 按启发式判断，仍能得到答案

def test_assess_failure_is_service_failure(self):       # EvidenceAssessment 脚本为 LLMError → status None，service_status failed，文本含 "service error"
def test_generate_failure_is_service_failure(self): ...
def test_retrieval_failure_is_service_failure(self):    # 用 FakeEmbedder(fail_queries=True)，并让 plan 返回中文检索词使 BM25 被跳过 → failed，assess 不被调用

def test_all_claims_unsupported_downgrades(self):       # 生成的结论只引用不存在的 ID → status unanswerable，trace 记录被丢掉的 ID

def test_chinese_question_uses_english_search_and_chinese_copy(self):
    # plan 返回 search_query "loud"；评估 partial、missing ["夜间"]；生成给出一条中文结论
    # 断言 search_queries[0] == "loud"，answer_text 含 "评论没有提到：夜间"

def test_no_leakage_into_any_prompt(self):
    # 跑一遍 answered 路径，再跑一遍 out_of_scope 路径
    for _, system, prompt in llm.calls:
        for secret in ("SECRET-ANSWER", "SECRET-SNIPPET"):
            self.assertNotIn(secret, system + prompt)

def test_bad_input(self):                               # 空问题 → ValueError；未知商品 → ProductNotFoundError
```

（`plan(...)`、`assess(...)` 是测试文件里构造 `QueryPlan`、`EvidenceAssessment` 的小函数。）

- [ ] **Step 2:** 运行 `REVIEWPROOF_TEST_DATABASE_URL=... .venv/bin/python -m unittest tests.test_workflow -b`，预期 FAIL。
- [ ] **Step 3: 实现 `state.py`、`nodes.py`、`graph.py`。** 用 `StateGraph(QAState)`、`START`、`END` 和 `add_conditional_edges` 建图，`compile()` 后 `invoke`。节点函数通过闭包或 `functools.partial` 拿到 `conn`、`llm`、`retrieval`、`max_rewrites`。
- [ ] **Step 4:** 运行同一命令，预期 PASS；全量测试 PASS。
- [ ] **Step 5: 提交** `feat: add LangGraph QA workflow with assessment, bounded rewrite and citation checks`

---

### Task 4: 问答命令行与真实数据验证

**Files:**
- Create: `reviewproof/workflow/cli.py`、`reviewproof/workflow/__main__.py`
- Modify: `CLAUDE.md`、`README.md`、`.env.example`
- Test: `tests/test_workflow_cli.py`

**Interfaces:**
- Consumes: `answer_question`、`RetrievalDeps`、`ClaudeLLM`、第 2 阶段 `reviewproof/retrieval/cli.py` 的 `default_embedder` 和 `default_reranker`。
- Produces: `main(argv=None, llm_factory=None, embedder_factory=None, reranker_factory=None) -> int`，命令 `python -m reviewproof.workflow`。

参数：`--product ASIN --question TEXT`，或 `--qid QID`。用 `--qid` 时只读 `question_text` 和 `product_id`，与检索命令的规则相同（商品不一致返回 2）。另有 `--no-rerank`、`--json`。

输出：
- 默认给人看，依次输出：答案文本；状态行 `status / service_status / verdict / route`；引用列表（编号、`chunk_id`、文本前 160 个字符）；trace（每个节点一行：状态、耗时、说明）；用量行（LLM 调用次数、输入和输出 token 数、总耗时）。
- `--json` 输出 `dataclasses.asdict(QAResult)`。
- 返回码：`service_status == failed` 返回 1；商品或题目不存在、问题为空返回 2；其余返回 0。
- `.env` 只在 `__main__.py` 里加载。

- [ ] **Step 1: 写失败测试 `tests/test_workflow_cli.py`**（数据库测试；注入 `FakeLLM`、`FakeEmbedder`、`FakeReranker`）：
  - answered 路径输出含 `[1]` 和 `status=answered`。
  - `--qid` 输出不含 `SECRET`。
  - 服务故障返回 1，输出含 `service error`。
  - 未知商品返回 2。
  - `--question "   "` 返回 2，不打印 traceback。
- [ ] **Step 2:** 运行，预期 FAIL。
- [ ] **Step 3: 实现。**
- [ ] **Step 4:** 运行，预期 PASS；全量测试 PASS。
- [ ] **Step 5: 真实数据验证。** 需要用户先在 `.env` 填好 `ANTHROPIC_API_KEY`，并同意产生少量 API 费用（Haiku 5.5 每题约 3–4 次调用）。用 dev 集跑下面 4 道题，把状态、引用、trace、token 数和耗时记进台账：
  - 一件 `small` 商品的题。
  - qid 17020（`large` 商品 B00007E7C8）。
  - 同一商品用中文问"音量可以调节吗？"。
  - 同一商品问 "How much does it cost?"，预期 `route = out_of_scope`。

  不要用这几次结果下质量结论，质量评测是第 5 阶段的事。
- [ ] **Step 6: 更新文档。**
  - `CLAUDE.md`：Commands 加问答命令；Architecture 加"Phase 3: QA workflow"一段（节点顺序、三态与 `service_status` 的关系、改写上限、防泄漏规则）。
  - `README.md`：当前进度加第 3 阶段，并给出一条问答命令示例。
  - `.env.example`：加 `REVIEWPROOF_LLM_MODEL`、`REVIEWPROOF_LLM_EFFORT`、`REVIEWPROOF_MAX_REWRITES` 的注释默认值。
- [ ] **Step 7: 提交** `feat: add QA command and document phase 3`
