# 第 5a 阶段：自动评测 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用一条命令、固定的数据和随机种子复跑评测，产出带 95% 置信区间和失败案例的报告。评测分三部分：
- 检索效果：BM25、向量、混合、混合加重排四种方案。
- 可答性：与 AmazonQA 银标对比，并跑两个对照基线。
- 成本与延迟。

本阶段不做人工金标（那是 5b），所有数字都标明"银标"及其局限。另外补上运行记录里缺的检索候选和分数（定位文档 F11）。

**Architecture:** 新增 `reviewproof/eval/` 包：
- `dataset.py`：按 split 读题，分层抽样。
- `alignment.py`：把 AmazonQA 的 `review_snippets` 对齐到评论块，作为银标参考证据。
- `retrieval_eval.py`：一次检索同时得到四种方案的排序，计算指标。
- `workflow_eval.py`：对抽样题跑完整工作流，以及 B0（不给证据）和 B1（全量上下文）两个基线；逐题写 JSONL，可以断点续跑。
- `metrics.py`：指标、bootstrap 置信区间、配对差值。
- `report.py`：生成 `metrics.json` 和中文 `report.md`。

命令是 `python -m reviewproof.eval`。工作流结果新增检索快照（候选、名次、分数），随运行记录写入 `qa_runs`，网页的"运行详情"可以查看。

**Tech Stack:** Python 3.10+、第 2–4 阶段的检索、工作流和运行记录、unittest；前端改动用 Vitest。不引入新的 Python 依赖（bootstrap 用标准库 `random` 和 `statistics`）。

**Spec:** [docs/product_positioning_cn.md](../../product_positioning_cn.md)：
- 5.2 节：片段偏向 BM25，银标只对片段成立。
- 5.5 节：防泄漏，只在 dev 调参。
- 第 6 节：指标、基线、报告规则。
- F11：运行记录回查。

以及 [docs/roadmap_cn.md](../../roadmap_cn.md) 第 3 节阶段 5。

## Global Constraints

- 在分支 `phase-5a-eval` 上工作，每个任务一次提交。
- 规模（用户已定）：Claude 调用只做小规模试跑，dev、test 各分层抽 50 题；检索评测不调用 Claude，默认跑整个 split（dev 239 题，其中混合检索商品 176 题）。
- **防泄漏：** 评测代码可以读 `review_snippets`、`answers` 和 `is_answerable_silver` 来算分，但送进工作流和基线的只能是 `question_text`。阈值、权重只在 dev 上看，test 只用来出最终数字。
- **措辞：** 所有报告和文档都要写明：
  - "可答性银标"是 AmazonQA 分类器基于 10 个 BM25 片段做的判断，换成完整评论后，部分"不可答"其实可答，所以对照银标的"误答率"会被高估。
  - "片段对齐的参考证据"来自 BM25 挑出的片段，检索对比天然偏向 BM25。
  - 两者都不是人工金标。
- 可答性按二值映射与银标对比：`answered`、`partial` 算"答了"，`unanswerable` 和不在服务范围算"拒答"。`service_status = failed` 的题不进入可答性分母，单独统计数量。
- 随机种子：抽样和 bootstrap 都用 `20261027`。bootstrap 重采样 1,000 次，取 2.5% 和 97.5% 分位数。
- 每份报告都要记下：git 提交号、`dataset_id` 和清单 SHA-256、向量模型、重排模型、LLM 模型、`RetrievalParams`、`max_rewrites`、种子、样本量、运行时间。
- 价格写成常量：Claude Haiku 5.5 每百万 token 输入 $0.10、输出 $0.50（提示词不超过 10 万 token 时）；超过 10 万 token 时为 $0.50 和 $2.50。报告里注明价格日期 2026-10-27。
- 输出目录 `reports/eval/<name>/`：`metrics.json`、`report.md`、`per_question.jsonl` 都提交进 git（试跑规模很小），`config.json` 也提交。
- 测试不调用真实 Claude 和模型，沿用 `FakeLLM`、`FakeEmbedder`、`FakeReranker`。

## Review Focus

1. **评测路径泄漏：** 评测代码手里有片段和社区回答，但它们绝不能出现在工作流或基线的任何提示词里。（任务 4 测试）
2. **试跑中途中断**（Ctrl-C、API 限流）：重跑同一命令，跳过已写入 JSONL 的题，不重复花钱；被中断的那题重新跑。（任务 4 测试）
3. **没有参考证据的题：** 片段一个都没对齐上，或题目是银标"不可答"时，不进入检索指标的分母，并在报告里写明各自有多少题。（任务 2、任务 3 测试）
4. **服务故障：** 某题 Claude 调用失败时，计入故障数，不计入可答性分母，也不能当成"拒答"算进正确拒答率。（任务 4、任务 5 测试）
5. **全量上下文商品：** 检索评测只对混合检索商品算指标；全量上下文商品单独列出数量，不和混合检索商品的数字混在一起。（任务 3 测试）

---

### Task 1: 运行记录补上检索候选与分数（F11）

**Files:**
- Modify: `reviewproof/workflow/graph.py`（`QAResult` 新增字段）、`reviewproof/workflow/nodes.py`、`reviewproof/workflow/state.py`、`frontend/src/api/types.ts`、`frontend/src/components/RunDetails.vue`
- Test: `tests/test_workflow.py`、`tests/test_runs.py`、`frontend/src/components/RunDetails.test.ts`

**Interfaces:**
- Produces（`state.py`）：

```python
@dataclass(frozen=True)
class RetrievalSnapshot:
    query: str
    strategy: str
    mode: str
    service_status: str
    channels: tuple[ChannelStatus, ...]     # 第 2 阶段的类型
    candidates: tuple[dict, ...]             # 每项 {"chunk_id", "review_id", "dense_rank", "dense_score",
                                             #       "bm25_rank", "bm25_score", "rrf_score", "rerank_score"}，不含文本
```

- `QAResult` 在末尾新增 `retrievals: tuple[RetrievalSnapshot, ...] = ()`：检索和改写重检索各追加一项。快照存的是最终证据 `candidates`（最多 8 条），不存整个 pool。检索失败时不追加。`state.py` 的 `retrieval_snapshots` 用 `operator.add` 累加。
- 网页：`RunDetails` 打开时调用 `api.run(runId)` 一次（同一个 run 只取一次），按检索次数分别显示候选表：名次、`chunk_id`、向量名次、BM25 名次、RRF 分数、重排分数。接口出错时显示"运行记录暂时取不到"。

- [ ] **Step 1: 写失败测试**
  - `test_workflow.py`：
    - `test_retrieval_snapshots_recorded`：answered 路径有 1 个快照，`candidates` 第一项是 `A-r00001-c00`，含 `dense_rank`。
    - `test_rewrite_adds_second_snapshot`：商品 C 的改写路径有 2 个快照，`query` 分别是 `"noise"` 和 `"battery"`。
  - `test_runs.py`：`test_snapshots_round_trip`，保存后取回的 `retrievals[0]["candidates"]` 不为空。
  - `RunDetails.test.ts`：
    - 展开后调用 `run` 并显示候选表。
    - `run` 出错时显示"运行记录暂时取不到"。
- [ ] **Step 2:** 运行，预期 FAIL。
- [ ] **Step 3: 实现。**
- [ ] **Step 4:** Python 全量测试和前端 `typecheck && test` 都 PASS。
- [ ] **Step 5: 提交** `feat: record retrieval candidates and scores in each run`

---

### Task 2: 评测题集、分层抽样与片段对齐

**Files:**
- Create: `reviewproof/eval/__init__.py`、`reviewproof/eval/dataset.py`、`reviewproof/eval/alignment.py`
- Test: `tests/test_eval_dataset.py`、`tests/test_eval_alignment.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class EvalQuestion:
    qid: str
    product_id: str
    category: str
    split: str
    strategy: str                 # full_context / hybrid（按 token_estimate 与阈值判断）
    question_text: str
    question_type: str            # yesno / descriptive（AmazonQA 原字段，只用于分组报告）
    answerable_silver: bool
    snippets: tuple[str, ...]

def load_questions(conn, split: str, full_context_max_tokens: int) -> list[EvalQuestion]    # 按 qid 排序
def sample_questions(questions: Sequence[EvalQuestion], n: int, seed: int) -> list[EvalQuestion]
def align_snippets(snippets: Sequence[str], chunks: Sequence[ChunkRow]) -> frozenset[str]   # 返回 chunk_id 集合
def silver_qrels(conn, questions: Sequence[EvalQuestion]) -> dict[str, frozenset[str]]       # qid → chunk_id 集合
```

规则：
- **抽样：** 按 `(category, answerable_silver)` 分层，按比例分配名额（向下取整，每层至少 1 题，再按余数从大到小补足 `n`），层内用 `random.Random(f"{seed}:{category}:{answerable}")` 抽取，结果按 `qid` 排序。`n` 大于等于题数时返回全部。
- **对齐：** 片段和块文本都先 `normalize_text`，再去掉全部空白字符做匹配，同时保留一份从"去空白后的位置"到"原位置"的映射。片段完整出现在某个块内时，命中这个块；片段跨越相邻两块时，两块都算。第 1 阶段审查实测发现，有 3.4% 的片段只有忽略空白才能对上（AmazonQA 的分词在标点两侧加了空格），这里统一处理。
- `silver_qrels` 只读当前题所属商品的块（按商品取，不跨商品）。

- [ ] **Step 1: 写失败测试**
  - `test_eval_dataset.py`（DatabaseTestCase）：
    - `test_load_questions_fields_and_strategy`
    - `test_sample_is_deterministic_and_stratified`：同样的种子两次结果一致；每层至少 1 题；`n` 大于题数时返回全部。
  - `test_eval_alignment.py`（离线）：
    - `test_exact_snippet_hits_chunk`
    - `test_whitespace_insensitive_match`：片段 `"great !! ! works"` 能对上块文本 `"great!!! works"`。
    - `test_html_entities_normalized`：片段里的 `&quot;` 能对上块里的 `"`。
    - `test_snippet_across_chunk_boundary_hits_both`
    - `test_unmatched_snippet_returns_empty`
    - `test_only_same_product_chunks`（DatabaseTestCase，`silver_qrels`）
- [ ] **Step 2:** 运行，预期 FAIL。
- [ ] **Step 3: 实现。**
- [ ] **Step 4:** PASS；全量测试 PASS。
- [ ] **Step 5: 提交** `feat: add eval question sampling and snippet-to-chunk alignment`

---

### Task 3: 检索评测

**Files:**
- Create: `reviewproof/eval/metrics.py`（本任务只放检索指标；bootstrap 在任务 5 补上）、`reviewproof/eval/retrieval_eval.py`
- Test: `tests/test_eval_metrics.py`、`tests/test_eval_retrieval.py`

**Interfaces:**

```python
# metrics.py
def recall_at_k(ranked: Sequence[str], relevant: AbstractSet[str], k: int) -> float
def hit_at_k(ranked, relevant, k) -> float
def mrr_at_k(ranked, relevant, k) -> float        # 第一个相关项名次的倒数，k 名以内没有则为 0
def ndcg_at_k(ranked, relevant, k) -> float       # 二值相关：DCG = Σ 1/log2(i+1)，IDCG 用 min(|relevant|, k) 个 1

# retrieval_eval.py
VARIANTS = ("bm25", "dense", "hybrid", "hybrid_rerank")

def rank_variants(conn, product_id: str, query: str, deps: RetrievalDeps) -> dict[str, list[str]]
def evaluate_retrieval(conn, questions: Sequence[EvalQuestion], qrels: Mapping[str, AbstractSet[str]],
                       deps: RetrievalDeps) -> list[dict]     # 每题一行
```

`rank_variants` 的规则：
- 每题只跑一次 `dense_search` 和 `bm25_cache.search`，各取 `channel_top_k` 条。
- `bm25` 和 `dense` 就是各自通道的排序。
- `hybrid` 是 `rrf_fuse` 的结果（不截断）。
- `hybrid_rerank` 把 `hybrid` 的前 `rerank_pool` 条按重排分数重新排序（同分时保持 RRF 顺序），后面接上 `hybrid` 剩下的部分。
- 查询词用原问题 `question_text`：检索评测衡量的是检索本身，不经过规划节点改写。

`evaluate_retrieval` 每题一行：`{"qid", "product_id", "category", "question_type", "strategy", "answerable_silver", "n_relevant", "included", "exclude_reason", "<variant>": {"recall@5", "recall@10", "recall@20", "recall@40", "hit@5", "mrr@10", "ndcg@5"}, "latency_ms"}`。
- `included` 为假的三种情况，`exclude_reason` 依次为：
  - `"full_context"`：商品走全量上下文；
  - `"silver_unanswerable"`：银标不可答；
  - `"no_aligned_snippet"`：没有对齐上的片段。

  这些题不计算指标。
- 某个通道出错时，那个方案记为 `null`，`exclude_reason` 照常为空。报告里统计各方案的出错次数。

- [ ] **Step 1: 写失败测试**
  - `test_eval_metrics.py`：给出几组手算的排序和期望值，分别检查 Recall、Hit、MRR、nDCG，包括 `relevant` 为空集时抛 `ValueError`。
  - `test_eval_retrieval.py`（DatabaseTestCase，商品 C 用 12 条评论、`full_context_max_tokens=0`）：
    - `test_variants_rank_relevant_chunk`：四种方案都返回排序，`hybrid_rerank` 把精确匹配的块排在第一。
    - `test_exclusions`：三种排除原因各有一题，`included` 为假。
    - `test_full_context_products_are_listed_not_scored`
- [ ] **Step 2:** 运行，预期 FAIL。
- [ ] **Step 3: 实现。**
- [ ] **Step 4:** PASS；全量测试 PASS。
- [ ] **Step 5: 提交** `feat: add retrieval evaluation across BM25, dense, hybrid and reranked variants`

---

### Task 4: 工作流评测与两个基线

**Files:**
- Create: `reviewproof/eval/workflow_eval.py`、`reviewproof/eval/baselines.py`
- Test: `tests/test_eval_workflow.py`

**Interfaces:**

```python
# baselines.py
class B0Answer(BaseModel):
    answered: bool                 # 模型认为自己能否回答
    answer: str

def b0_no_evidence(llm: LLM, question: str, category: str) -> LLMResult[B0Answer]
    # 只给品类和问题，不给任何评论；系统提示：可以根据一般知识回答，确实无法判断时把 answered 设为 false

# workflow_eval.py
SYSTEMS = ("workflow", "b0_no_evidence", "b1_full_context")

def run_workflow_eval(conn, questions: Sequence[EvalQuestion], *, llm: LLM, deps: RetrievalDeps,
                      max_rewrites: int, systems: Sequence[str], out_path: Path) -> list[dict]
```

规则：
- 逐题、逐系统运行，每完成一个 `(qid, system)` 就追加一行 JSON 到 `out_path`，并立刻 `flush`。
- 启动时先读 `out_path`，跳过已有的 `(qid, system)`，实现断点续跑；文件末尾残缺的一行（被中断时写了一半）要丢弃。
- `workflow`：调用 `answer_question(conn, product_id, question_text, ...)`。
- `b1_full_context`：同样调用 `answer_question`，但 `deps.params` 改为 `full_context_max_tokens = 10**9`，强制把全部评论放进上下文。只对混合检索商品运行（全量上下文商品的 B1 和 `workflow` 完全相同）。
- `b0_no_evidence`：调用 `b0_no_evidence`。
- 每行字段：`{"qid", "system", "product_id", "category", "question_type", "strategy", "answerable_silver", "status", "service_status", "route", "verdict", "answered_binary", "n_citations", "citation_out_of_product", "llm_calls", "input_tokens", "output_tokens", "cost_usd", "latency_ms", "rewrites_used", "retrieval_modes", "answer_text", "error"}`。
  - `answered_binary`：`workflow` 和 B1 取 `status in ("answered", "partial")`；B0 取 `answered`。服务故障时为 `null`。
  - `citation_out_of_product`：引用里属于其他商品的块数，预期永远为 0。
  - `cost_usd`：工作流结果只有整题的 token 总数，没有每次调用的数，所以按"整题输入 token / 调用次数"得出平均每次的输入量。平均不超过 10 万时用 $0.10 / $0.50，超过时用 $0.50 / $2.50；再用整题的输入、输出 token 总数乘以对应单价。`config.json` 里注明这是近似值。
  - `error`：工作流抛出意外异常时，记录异常文本，`service_status` 记为 `failed`，然后继续下一题，不中断整个评测。

- [ ] **Step 1: 写失败测试 `tests/test_eval_workflow.py`**（DatabaseTestCase；`FakeLLM` 用可调用对象，按提示词内容返回结果）：
  - `test_rows_written_per_system`：三个系统各写一行，字段齐全，B1 只出现在混合检索商品。
  - `test_resume_skips_done_and_drops_partial_line`：先写入一行完整记录和半行残缺记录，运行后完整那行没有重复，残缺那行被重跑。
  - `test_service_failure_row`：评估节点抛 `LLMError` → `answered_binary` 为 `null`，`service_status` 为 `failed`。
  - `test_unexpected_exception_is_recorded_and_run_continues`
  - `test_no_snippet_or_answer_leaks_into_prompts`：题目带 `SECRET-SNIPPET`，社区回答带 `SECRET-ANSWER`，三个系统所有调用的提示词里都没有这两个字符串。
  - `test_cost_uses_haiku_prices`：20 次调用、输入 1,000,000 token、输出 0 → 平均每次 5 万 token，成本 0.10 美元；1 次调用、输入 200,000 token、输出 1,000,000 token → 用长提示词那一档，成本 0.10 + 2.50 = 2.60 美元。
- [ ] **Step 2:** 运行，预期 FAIL。
- [ ] **Step 3: 实现。**
- [ ] **Step 4:** PASS；全量测试 PASS。
- [ ] **Step 5: 提交** `feat: add workflow evaluation with B0 and B1 baselines and resumable output`

---

### Task 5: 指标汇总、置信区间与报告

**Files:**
- Modify: `reviewproof/eval/metrics.py`
- Create: `reviewproof/eval/report.py`
- Test: `tests/test_eval_report.py`

**Interfaces:**

```python
# metrics.py
def bootstrap_ci(values: Sequence[float], *, seed: int = 20261027, resamples: int = 1000) -> tuple[float, float, float]
    # (均值, 2.5% 分位, 97.5% 分位)；values 为空时抛 ValueError
def paired_diff_ci(a: Sequence[float], b: Sequence[float], *, seed=20261027, resamples=1000) -> tuple[float, float, float]
    # 同一组重采样下标同时作用于 a 和 b，返回 mean(a - b) 及其区间；长度不同抛 ValueError
def answerability_metrics(rows: Sequence[dict]) -> dict
    # 只用 answered_binary 不为 null 的行：
    # correct_refusal_rate = P(拒答 | 银标不可答)，false_refusal_rate = P(拒答 | 银标可答)，
    # answer_precision / answer_recall / answer_f1（以"答了"为正类、银标可答为真值），accuracy，
    # 每个率都带 bootstrap 区间；另报告 n、n_service_failed、status 三态分布

# report.py
def build_report(*, retrieval_rows: Sequence[dict] | None, workflow_rows: Sequence[dict] | None,
                 config: dict, out_dir: Path) -> dict      # 写 metrics.json 和 report.md，返回 metrics
```

`report.md` 的结构（中文）：
1. **概要：** 数据和版本信息；银标与片段偏差的说明（照抄 Global Constraints 里的措辞）。
2. **检索：**
   - 各方案指标表，每格为 `均值 [下界, 上界]`，样本是 `included` 为真的题。
   - `hybrid_rerank` 对比其余三种方案的 Hit@5 和 nDCG@5 配对差值。
   - 排除各原因的题数，以及各方案出错的次数。
   - 按品类细分 Hit@5。
3. **可答性：** 每个系统一张指标表、三态分布、服务故障数，以及"服务故障被报告成评论无法确认"的次数（按构造应为 0，照样统计）。
4. **成本与延迟：** 每个系统的每题平均成本、P50 和 P95 延迟、LLM 调用次数，以及检索通道降级率（来自 `retrieval_modes` 中带降级的比例）。
5. **失败案例：** 每类最多列 5 题，显示问题、系统状态、银标和答案前 200 字：
   - 误拒（银标可答但系统拒答）；
   - 对照银标的误答（银标不可答但系统答了）。由于银标的局限，要注明"需人工核对"。
6. **局限与下一步：** 指向 5b 的人工金标。

不写任何提升百分比的结论，只列数字和区间。

- [ ] **Step 1: 写失败测试 `tests/test_eval_report.py`**（离线，构造假的行数据）：
  - `test_bootstrap_ci_deterministic_and_bounds`：同样的种子结果相同；全是 1 时区间为 `(1, 1, 1)`。
  - `test_paired_diff_uses_same_resamples`：`a == b` 时差值区间为 `(0, 0, 0)`。
  - `test_answerability_metrics_excludes_service_failures`：手算的拒答率、F1，以及 `n_service_failed`。
  - `test_report_files_and_sections`：生成的 `report.md` 含六个小节标题、"银标"字样和"需人工核对"；`metrics.json` 可以被解析，并含 `config`。
  - `test_report_without_workflow_rows`：只有检索结果时也能生成报告，可答性一节写"本次未运行"。
- [ ] **Step 2:** 运行，预期 FAIL。
- [ ] **Step 3: 实现。**
- [ ] **Step 4:** PASS；全量测试 PASS。
- [ ] **Step 5: 提交** `feat: add evaluation metrics with bootstrap intervals and Markdown report`

---

### Task 6: 评测命令、试跑与文档

**Files:**
- Create: `reviewproof/eval/cli.py`、`reviewproof/eval/__main__.py`（只在这里加载 `.env`）
- Create（试跑产物）：`reports/eval/pilot-dev/`、`reports/eval/pilot-test/`
- Modify: `.gitignore`（确认 `reports/` 没被忽略）、`CLAUDE.md`、`README.md`
- Test: `tests/test_eval_cli.py`

**Interfaces:**
- `python -m reviewproof.eval --split dev|test --name NAME [--sample N] [--retrieval-only] [--systems workflow,b0_no_evidence,b1_full_context] [--seed 20261027]`
  - 检索评测跑整个 split。
  - 工作流评测跑 `--sample` 道抽样题（默认 50），`--retrieval-only` 时跳过。
  - 输出写到 `reports/eval/NAME/`：`config.json`、`retrieval.jsonl`、`per_question.jsonl`（工作流结果）、`metrics.json`、`report.md`。
  - 同名目录已存在时续跑：检索结果重新计算（不花钱），工作流结果按断点续跑处理。
  - 进度打到 stderr，每题一行。
- `main(argv, llm_factory=None, embedder_factory=None, reranker_factory=None) -> int`。

- [ ] **Step 1: 写失败测试 `tests/test_eval_cli.py`**（DatabaseTestCase；注入 Fake 组件，输出到临时目录）：
  - `test_end_to_end_small`：生成 5 个文件，`report.md` 含"检索"和"可答性"。
  - `test_retrieval_only_makes_no_llm_calls`
  - `test_rerun_resumes`：第二次运行时 `FakeLLM` 的脚本为空，也不抛出"脚本用完"，说明没有重跑已完成的题。
- [ ] **Step 2:** 运行，预期 FAIL。
- [ ] **Step 3: 实现。**
- [ ] **Step 4:** PASS；全量测试 PASS。
- [ ] **Step 5: 试跑。** 会调用真实 Claude，**先征得用户同意**。估计每个 split 50 题，三个系统合计约 300 次 Haiku 调用，B1 的提示词较长，费用约 1–2 美元。重排在 CPU 上较慢，每个 split 大约要 30–60 分钟，放到后台运行：
  - 先跑 `--split dev --name pilot-dev`，看过报告确认没问题后，再跑 `--split test --name pilot-test`。
  - 把两份报告的关键数字、总费用和总耗时记进台账。
  - 不在任何地方写超出报告数字的结论。
- [ ] **Step 6: 更新文档。**
  - `CLAUDE.md`：Commands 加评测命令；Architecture 加"Phase 5a: evaluation"一段（模块、指标定义、二值映射、银标与片段偏差的说明、断点续跑）。
  - `README.md`：当前进度加 5a，指向两份试跑报告。
- [ ] **Step 7: 提交** `feat: add evaluation command and pilot reports`
