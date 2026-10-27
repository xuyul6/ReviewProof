# 第 5b 阶段：人工金标 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 做出约 200 题的人工金标，并用同一条评测命令按金标出报告。金标包括三项：
- 可答性三态（能回答 / 部分 / 不能）；
- 每个候选评论块的证据等级；
- 是非题的结论。

做法是"子 agent 预标、人逐题审核"：每一题都经过人确认才算金标；20% 的题先由人盲标，用来算人和子 agent 之间的一致率。

**Architecture:**
- 新增 `reviewproof/annotate/` 包：
  - `store.py`：标注的数据结构、校验和读写。
  - `pool.py`：抽题、构造候选池、产品内搜索。
  - `cli.py`：`python -m reviewproof.annotate` 的各个子命令。
- 数据库新增两张表：`annotation_items`（标注题和候选池）、`annotation_labels`（预标、盲标、终审三种来源）。
- 终审结果导出成 `annotations/<label_set>/labels.jsonl` 提交进 git。评测只读这个文件，不读数据库，所以从 git 就能复跑。
- 审核页面放在现有的 Vue 应用里（`#/annotate`），接口只在 `REVIEWPROOF_ANNOTATION=1` 时注册。
- 评测：`python -m reviewproof.eval` 加 `--labels <labels.jsonl>`。给了这个参数，就用金标代替银标算检索和可答性，并多出三项：三态混淆矩阵、是非题结论准确率、一致率。

**Tech Stack:** Python 3.10+、psycopg 3、FastAPI、unittest；Vue 3 + Vitest。不引入新依赖：Cohen's kappa 用标准库手算。

**Spec:** [docs/product_positioning_cn.md](../../product_positioning_cn.md)：
- 5.6 节：人工标注。
- 第 6 节：指标、基线（含 B2：AmazonQA 自带分类器）、报告规则。

以及 [docs/roadmap_cn.md](../../roadmap_cn.md) 第 3 节阶段 5、阶段 6，和 [第 5a 阶段计划](2026-10-27-phase-5a-evaluation.md)：沿用它的抽样、检索评测和工作流评测代码。

## Global Constraints

- 在分支 `phase-5b-gold` 上工作，每个任务一次提交。
- **标注集 `gold_v1`：** 用户已定，dev 100 题 + test 100 题。
  - 每个 split 按 品类 × 题型 × 银标正负 分层抽样，种子 `20261027`。
  - 每个 split 中 20% 的题（各 20 题）设为盲标题。
  - 与 5.6 节"从 test 抽约 200 题"不同：第 6 阶段要在金标 dev 上定阈值，所以两个 split 各一半。题数是命令参数，可以改。
- **标注规范版本 `gold-v1`：** 写在 `docs/annotation_guidelines_cn.md`。规范的正文以本计划任务 1 为准，每条标注都记下规范版本。
- **证据等级：**
  - `2` = 单独读这一块，就能支撑回答或回答的一部分；
  - `1` = 相关，但不足以支撑；
  - `0` = 无关。
  - 检索指标里"相关"指等级 2；nDCG 按等级加权（增益等于等级）。
- **候选池：**
  - 全量上下文商品：该商品的全部评论块。
  - 混合检索商品：bm25、dense、hybrid、hybrid_rerank 四种排序各取前 10，合并去重。检索参数用 `RetrievalParams` 的默认值。
  - 候选池按 `chunk_id` 排序展示，不暴露是哪个方案排在前面。
  - 标注者用产品内搜索找到的块也可以给等级，记为池外块。
- **校验规则**（预标和终审都要满足；盲标只有可答性和结论两项）：
  - 可答性 ∈ {answered, partial, unanswerable}。
  - 题型是 `yesno` 时，结论 ∈ {yes, no, mixed, unknown}，并且"结论 = unknown"当且仅当"可答性 = unanswerable"；其他题型的结论必须为空。
  - 候选池里每个块都要有等级；池外块只能是该商品的块。
  - answered 和 partial 至少有一个等级 2 的块；unanswerable 不能有等级 2 的块（5.6 节："不可答题不编造相关 ID"）。
  - 理由不能为空。
- **防泄漏：**
  - 预标材料（`packet` 子命令的输出）只包含问题文本、品类、题型和候选池的评论块，不包含银标、片段和社区回答。
  - 审核页面可以给人看社区回答，但要折叠起来并注明"弱参考，判断以评论为准"；不展示银标和片段。
  - 金标文件和标注表都不进入任何提示词。
- **盲标：**
  - 盲标题在提交盲标之前，接口不返回预标内容，列表接口也不返回。
  - 盲标提交后不能修改。
  - 盲标题必须先有盲标，才能终审。
- **金标文件：** 每一行都带 `label_set`、`dataset_id`、`guideline_version`、`annotator`、候选池，以及预标和盲标的快照（用来算一致率）。
  - 评测读入时，`dataset_id` 与当前数据库不一致、或有块已不存在，就拒绝运行，退出码 2。
- **标注表不随重新导入删除：** `annotation_items.qid` 引用 `questions`，不加 `ON DELETE CASCADE`。重新导入如果要删掉已标注的题，会报错，而不是悄悄删掉人工标注。
- **预标用子 agent：**
  - 用 Claude Code 的子 agent（Opus），每批 10 题。走会话额度，不用 `.env` 里的 API key。
  - 子 agent 只能用 `packet` 和 `search` 两个子命令读数据，结果写成 JSONL，再用 `import-drafts` 导入。
  - 子 agent 和被测系统（Haiku）属于同一个模型家族，所以每一题都必须经过人审核。
- **金标评测要调用真实 Claude：** 开跑前先征得用户同意。预计两个 split 合计约 $4–5，主要花在 dev 那件 22 万 token 商品的 B1 上。
- 测试不调用真实 Claude 和模型，沿用 `FakeLLM`、`FakeEmbedder`、`FakeReranker`。
- **本计划不包括：**
  - 结论支持率（拆原子结论、核对引用）；
  - 中文提问小题集；
  - 第 5a 阶段发现的三处工作流问题；
  - 是非题结论与社区多数回答的一致率（社区回答是自由文本，要先做结论抽取）。

  这四项另起计划。

## Review Focus

1. **重新导入数据时碰到已标注的题：** 必须报错，不能级联删除人工标注。（任务 1 测试）
2. **盲标题的预标泄漏：** 盲标提交前，单题接口和列表接口都不能返回预标的可答性、结论或等级。（任务 3 测试）
3. **预标材料泄漏：** `packet` 的输出里不能出现银标、片段或社区回答。（任务 2 测试）
4. **不合规的标注：** 其他商品的块、不存在的块、候选池里漏标的块、unanswerable 却有等级 2 的块，都要被拒绝并说明原因，而且不写入任何部分结果。（任务 1、任务 2 测试）
5. **金标文件和数据库对不上：** `dataset_id` 不一致，或引用的块已不存在时，评测拒绝运行，不悄悄把这些块当成不相关。（任务 5 测试）

---

### Task 1: 标注规范、表和读写

**Files:**
- Create: `docs/annotation_guidelines_cn.md`
- Modify: `reviewproof/schema.sql`、`reviewproof/db.py`（`ALL_TABLES` 加上两张新表）
- Create: `reviewproof/annotate/__init__.py`、`reviewproof/annotate/store.py`
- Test: `tests/test_annotate_store.py`

**Interfaces:**
- Produces:
  - `GUIDELINE_VERSION = "gold-v1"`、`ANSWERABILITY`、`VERDICTS`、`GRADES = (0, 1, 2)`
  - `@dataclass(frozen=True) ItemSpec(qid: str, split: str, blind: bool, pool: tuple[str, ...])`
  - `@dataclass(frozen=True) Label(qid: str, source: str, answerability: str, verdict: str | None, grades: Mapping[str, int], rationale: str, annotator: str)`，`source` ∈ {draft, blind, final}
  - `class LabelError(ValueError)`
  - `add_items(conn, label_set: str, items: Sequence[ItemSpec]) -> None`：标注集已存在时抛 `LabelError`。
  - `list_items(conn, label_set) -> list[dict]`：每项含 qid、split、blind、status。status 只有三种：pending（没有预标）、drafted（有预标没终审）、done（已终审）。不含任何标注内容。
  - `get_item(conn, label_set, qid) -> dict | None`：
    - 题目信息：qid、product_id、category、split、question_text、question_type、blind；
    - `pool`：块列表，每块是 {chunk_id, review_id, text}；
    - `labels`：{source: Label 的字典}，只含已有的来源。
  - `save_label(conn, label_set, label: Label) -> None`：先校验，再按 (label_set, qid, source) 写入或覆盖。
    - 盲标已存在时，再存盲标会抛 `LabelError`。
    - 盲标题没有盲标就存终审，也会抛 `LabelError`。
  - `export_labels(conn, label_set) -> list[dict]`：只导出已终审的题，按 qid 排序。每行字段：
    - 标识：label_set、dataset_id、guideline_version、qid、product_id、split、question_type；
    - 终审内容：answerability、verdict、grades、rationale、annotator、pool；
    - 快照：`draft`（{answerability, verdict, grades} 或 null）、`blind`（{answerability, verdict} 或 null）；
    - `changed_from_draft: bool`：可答性、结论或任一等级与预标不同就为 true。

- [ ] **Step 1: 写标注规范 `docs/annotation_guidelines_cn.md`**（中文，标题注明版本 gold-v1），必须包含：
  - **判断依据：** 只看该商品的评论；不用常识，不看社区回答。评论是西班牙语等其他语言的，同样算数。
  - **answered：** 评论里有直接回答问题核心的内容；一位买家的明确陈述就够，写理由时按"有买家说"转述。意见不一时也算 answered，结论记为 mixed。
  - **partial：** 问题有几个部分，只答了其中一部分；或者只有相近情形的信息，比如问 2012 款能不能装，评论说 2013 款装上了。
  - **unanswerable：** 评论没有提到。问题本身超出评论范围时（保修条款、官方尺寸、去哪里买、和别的商品比较），如果评论里有相关的亲身经历，就按内容判 partial 或 answered；没有才判 unanswerable。
  - **证据等级**的 0、1、2 定义，各配一个例子。
  - **结论**的规则，见 Global Constraints。
  - **理由：** 一两句话，说明依据哪几块。
  - **找证据：** 候选池里没有答案、但怀疑评论里有时，先用搜索找；找到的块也给等级。
- [ ] **Step 2: 写失败的测试**（`tests/test_annotate_store.py`，`DatabaseTestCase`；用 `make_manifest`、`make_product` 造 A、B 两件商品，A 有一道 `yesno` 题 qa 和一道 `descriptive` 题 qd）

```python
def test_save_and_export_round_trip(self):
    # draft then final for qa; export has draft snapshot, changed_from_draft False when identical
def test_validation_rejects_and_writes_nothing(self):
    for bad in (grade on "B-r00000-c00", grade on "A-r09999-c00", pool chunk missing a grade,
                unanswerable with a grade-2 chunk, answered without grade 2, yesno without verdict,
                descriptive with verdict, verdict "unknown" with answered, blank rationale):
        with self.assertRaises(LabelError): save_label(...)
    self.assertEqual(get_item(...)["labels"], {})
def test_blind_flow(self):
    # blind item: final before blind → LabelError; blind saved; second blind → LabelError; final ok
def test_list_items_status_and_no_label_content(self):
    # statuses pending/drafted/done; dict keys exactly {"qid", "split", "blind", "status"}
def test_reingest_cannot_cascade_delete_annotated_question(self):
    # add_items for qa, then load_dataset without qa → raises psycopg.errors.ForeignKeyViolation; label still there
def test_add_items_twice_refused(self):
```

- [ ] **Step 3:** 运行 `REVIEWPROOF_TEST_DATABASE_URL=... python -m unittest tests.test_annotate_store`。Expected：导入失败（`reviewproof.annotate.store` 不存在）。
- [ ] **Step 4: 实现。** 要点：
  - **表结构：**
    - `annotation_items(label_set TEXT, qid TEXT REFERENCES questions, dataset_id TEXT NOT NULL, split TEXT NOT NULL, blind BOOLEAN NOT NULL, pool JSONB NOT NULL, created_at TIMESTAMPTZ DEFAULT now(), PRIMARY KEY (label_set, qid))`
    - `annotation_labels(label_set, qid, source TEXT CHECK (source IN ('draft','blind','final')), answerability TEXT NOT NULL, verdict TEXT, grades JSONB NOT NULL, rationale TEXT NOT NULL, annotator TEXT NOT NULL, guideline_version TEXT NOT NULL, created_at TIMESTAMPTZ DEFAULT now(), PRIMARY KEY (label_set, qid, source), FOREIGN KEY (label_set, qid) REFERENCES annotation_items)`
  - **`ALL_TABLES`** 把两张新表放在 `questions` 前面。
  - **校验**放在一个纯函数 `validate_label(label, *, pool, product_chunk_ids, question_type)` 里：盲标跳过等级和理由的校验，其余按 Global Constraints 的规则。
- [ ] **Step 5:** 跑同一条命令，Expected：全部通过；再跑全量测试，Expected：OK。
- [ ] **Step 6: 提交** `feat: add gold annotation guidelines, tables and label store`

### Task 2: 抽题、候选池和标注命令

**Files:**
- Modify: `reviewproof/eval/dataset.py`（`sample_questions` 增加参数 `by`）
- Create: `reviewproof/annotate/pool.py`、`reviewproof/annotate/cli.py`、`reviewproof/annotate/__main__.py`
- Test: `tests/test_annotate_cli.py`；`tests/test_eval_dataset.py` 补测试

**Interfaces:**
- Consumes：任务 1 的 store；5a 的 `load_questions`、`rank_variants`、`VARIANTS`。
- Produces:
  - `sample_questions(questions, n, seed, by: Sequence[str] = ("category", "answerable_silver"))`：默认行为不变，5a 的测试要照样通过。
  - `POOL_DEPTH = 10`
  - `build_pool(conn, q: EvalQuestion, deps: RetrievalDeps) -> tuple[str, ...]`：按 `chunk_id` 排序。全量上下文商品返回全部块；混合检索商品返回各方案前 10 的并集，跳过为 None 的方案。四个方案全部失败时抛 `RuntimeError`。
  - `pick_items(questions_by_split: Mapping[str, Sequence[EvalQuestion]], counts: Mapping[str, int], *, blind_fraction: float, seed: int) -> list[tuple[EvalQuestion, bool]]`：
    - 分层键是 `("category", "question_type", "answerable_silver")`；
    - 每个 split 用 `Random(f"{seed}:blind:{split}")` 选出 `round(n * blind_fraction)` 道盲标题。
  - `search_product(conn, product_id: str, query: str, bm25_cache: BM25Cache, *, exclude: AbstractSet[str], limit: int = 20) -> list[dict]`：
    - 结果是 BM25 前 `limit` 名加上大小写不敏感的子串匹配，去重，排除 `exclude` 里的块；
    - 每项是 {chunk_id, review_id, text}。
  - 命令 `python -m reviewproof.annotate`：
    - `pick --set gold_v1 --dev 100 --test 100 [--blind-fraction 0.2] [--seed 20261027] [--out-root annotations]`：写入数据库，并写出 `annotations/gold_v1/items.jsonl`。每行含 qid、split、blind、pool、dataset_id、pool_depth、retrieval_params、git_commit。标注集已存在时退出码 2。
    - `packet --set gold_v1 --qids a,b,...`：向 stdout 输出 Markdown。每题依次是 qid、品类、题型、问题，然后每块一行 `[chunk_id] 文本`，最后附上 Step 3 的 JSONL 格式说明。
    - `search --set gold_v1 --qid X --query TEXT`：输出池外的匹配块。
    - `import-drafts --set gold_v1 FILE...`：
      - 每行是 {qid, answerability, verdict, grades, rationale, annotator}，`source` 固定记为 draft；
      - 逐行校验，合规的写入，不合规的打印 `qid: 原因`；
      - 有被拒的行时退出码 1。
    - `status --set gold_v1`：按 split 统计 pending、drafted、done 的数量，以及盲标题已盲标的数量。
    - `export --set gold_v1 [--out-root annotations]`：写出 `annotations/gold_v1/labels.jsonl`。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_eval_dataset.py
def test_sample_by_question_type_covers_every_stratum(self):
# tests/test_annotate_cli.py (DatabaseTestCase; FakeEmbedder/FakeReranker via factories as in test_eval_cli)
def test_pick_writes_items_with_pools_and_blind_share(self):
    # pools sorted, full-context product pool == all its chunks; blind count == round(n*0.2) per split
def test_pick_twice_exits_2(self):
def test_packet_has_question_and_pool_but_no_silver_snippet_or_answer(self):
    # SECRET-SNIPPET / SECRET-ANSWER / "is_answerable" / "silver" absent; every pool chunk_id present
def test_import_drafts_reports_rejects_and_keeps_good_lines(self):
    # one good line, one with a chunk of another product → exit 1, good one stored, message names the qid
def test_search_excludes_pool_and_other_products(self):
def test_export_writes_only_final_labels(self):
```

- [ ] **Step 2:** 运行两个测试文件。Expected：FAIL（`annotate` 模块和 `by` 参数都还不存在）。
- [ ] **Step 3: 实现。** `cli.py` 按 5a `eval/cli.py` 的写法：`main(argv, embedder_factory, reranker_factory)` 可注入，`.env` 只在 `__main__.py` 里加载。`packet` 末尾附的格式说明全文如下：

```
每题输出一行 JSON：{"qid": "...", "answerability": "answered|partial|unanswerable", "verdict": "yes|no|mixed|unknown|null", "grades": {"<chunk_id>": 0|1|2, ...}, "rationale": "...", "annotator": "agent:<model>"}
候选池里的每个块都要给等级；搜索找到的池外块可以加进 grades。规则见 docs/annotation_guidelines_cn.md（gold-v1）。
```

- [ ] **Step 4:** 跑测试，Expected：通过；全量测试，Expected：OK。
- [ ] **Step 5: 提交** `feat: add annotation sampling, candidate pools and annotate command`

### Task 3: 审核接口

**Files:**
- Modify: `reviewproof/config.py`（增加 `annotation_enabled: bool = False`，读 `REVIEWPROOF_ANNOTATION`；增加 `annotator: str = "human"`，读 `REVIEWPROOF_ANNOTATOR`）
- Create: `reviewproof/api/annotation.py`（`APIRouter`）
- Modify: `reviewproof/api/app.py`（`annotation_enabled` 为真时才 `include_router`）
- Test: `tests/test_api_annotation.py`

**Interfaces:**
- Consumes：任务 1 的 store；任务 2 的 `search_product`。
- Produces（都以 `/v1/annotation/{label_set}` 为前缀）：
  - `GET /items` → `{"items": list_items(...)}`
  - `GET /items/{qid}`：返回 `get_item`，再加两样东西：`community_answers`（回答文本的列表）和 `draft_hidden: bool`。盲标题没有盲标时，`labels` 里去掉 draft，`draft_hidden` 为 true。题不存在返回 404。
  - `POST /items/{qid}/blind`：请求体 {answerability, verdict}。返回值同 GET。
    - 不是盲标题，或者已经盲标过，返回 409。
  - `PUT /items/{qid}/final`：请求体 {answerability, verdict, grades, rationale}，annotator 取设置里的值。返回值同 GET。
    - 校验失败返回 422，`detail` 写明原因；
    - 盲标题还没盲标，返回 409。
  - `GET /items/{qid}/search?q=`：返回 `{"chunks": search_product(...)}`。
  - 返回内容不包含 `is_answerable_silver` 和 `review_snippets`。

- [ ] **Step 1: 写失败的测试**

```python
def test_routes_absent_when_disabled(self):            # 404 on /v1/annotation/gold_v1/items
def test_blind_item_hides_draft_until_blind_label(self):
    # GET: draft_hidden True, "draft" not in labels and "SECRET-DRAFT-RATIONALE" not in response text;
    # POST blind → draft visible; second POST → 409
def test_final_validation_error_is_422_with_reason(self):
def test_final_before_blind_is_409(self):
def test_item_never_contains_silver_or_snippets(self):  # "SECRET-SNIPPET", "is_answerable" absent
def test_search_returns_out_of_pool_chunks(self):
```

- [ ] **Step 2:** 运行。Expected：FAIL。
- [ ] **Step 3:** 实现。数据库连接沿用 `Services.connect()`；搜索用 `Services` 检索时用的那个 `BM25Cache`，在进程锁下使用，因为它不是线程安全的。
- [ ] **Step 4:** 跑测试，Expected：通过；全量测试，Expected：OK。
- [ ] **Step 5: 提交** `feat: add annotation review API behind REVIEWPROOF_ANNOTATION`

### Task 4: 审核页面

**Files:**
- Modify: `frontend/src/main.ts`（`location.hash` 以 `#/annotate` 开头时挂载 `AnnotateApp`）
- Create: `frontend/src/api/annotation.ts`（客户端和类型）、`frontend/src/annotate/AnnotateApp.vue`、`frontend/src/annotate/ChunkGrader.vue`
- Test: `frontend/src/annotate/AnnotateApp.test.ts`、`frontend/src/api/annotation.test.ts`

**Interfaces:**
- Consumes：任务 3 的接口。
- Produces：页面 `http://localhost:8000/#/annotate?set=gold_v1`，`set` 默认是 gold_v1。

页面行为：
- **左栏：**
  - 题目列表，可按 split 和状态筛选，显示 已完成/总数。
  - "下一道待审"按钮，跳到第一道不是 done 的题。
- **盲标题、还没盲标时：** 只显示问题、候选池的块（不显示等级）、三个可答性按钮；是非题再加结论选择。点"提交盲标"后，显示预标，进入审核。
- **审核时：**
  - 可答性和结论预先选成预标的值。
  - 块列表：预标等级 ≥1 的块排在前面；等级 0 的块折叠成"其余 N 块（预标为无关）"，可以展开。每块有 0、1、2 三个按钮。
  - 搜索框：查到的池外块可以给等级，加进标注。
  - 理由输入框，预先填上预标的理由。
  - 社区回答折叠起来，标题是"社区回答（弱参考，判断以评论为准）"。
- **快捷键：** 1、2、3 选可答性；Ctrl+Enter 保存并跳到下一道待审。
- **出错时：** 422 和 409 的 `detail` 显示在保存按钮旁边，不跳题。

- [ ] **Step 1: 写失败的测试**（Vitest，用假的客户端）

```ts
it("hides the draft until the blind label is submitted")
it("prefills from the draft and saves changed grades and answerability")
it("shows the 422 detail and stays on the item")
it("next pending skips done items")
it("adds a searched out-of-pool chunk to the saved grades")
```

- [ ] **Step 2:** `cd frontend && npm test`。Expected：FAIL。
- [ ] **Step 3:** 实现。样式沿用 `style.css` 的变量，不新增依赖。
- [ ] **Step 4:** `npm run typecheck && npm test && npm run build`。Expected：全部通过。
- [ ] **Step 5: 手动检查。** 用测试数据库起 API（设 `REVIEWPROOF_ANNOTATION=1`），在浏览器里走一遍盲标和审核两个流程，确认没有控制台错误。
- [ ] **Step 6: 提交** `feat: add annotation review page`

### Task 5: 按金标评测

**Files:**
- Create: `reviewproof/eval/gold.py`
- Modify: `reviewproof/eval/metrics.py`、`reviewproof/eval/retrieval_eval.py`、`reviewproof/eval/workflow_eval.py`、`reviewproof/eval/report.py`、`reviewproof/eval/cli.py`
- Test: `tests/test_eval_gold.py`；`tests/test_eval_metrics.py`、`tests/test_eval_report.py`、`tests/test_eval_cli.py` 补测试

**Interfaces:**
- Consumes：任务 1 的导出格式。
- Produces:
  - `@dataclass(frozen=True) GoldLabel(qid, split, product_id, question_type, answerability, verdict, grades: Mapping[str, int], draft: dict | None, blind: dict | None)`
  - `class GoldError(ValueError)`
  - `load_gold(conn, path: Path) -> dict[str, GoldLabel]`：文件不存在、`dataset_id` 和数据库里的不一致、或者引用的块已不存在，都抛 `GoldError`。
  - `gold_qrels(labels) -> dict[str, dict[str, int]]`：只含 answered 和 partial 的题，值是 {chunk_id: 等级}。
  - `metrics.cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float | None`：长度为 0、或两边各自只出现一个类别时返回 None。
  - `metrics.ndcg_graded_at_k(ranked, gains: Mapping[str, int], k) -> float`
  - `metrics.answerability_metrics(rows, label_key: str = "answerable_silver")`：默认行为不变。
  - `metrics.three_state(rows) -> dict`：只统计 `service_status != failed` 且 `answerability_gold` 不为空的行。内容是 3×3 混淆矩阵 `{gold: {pred: n}}`、三态一致率（带 bootstrap 区间），以及 `n`。
  - `evaluate_retrieval(conn, questions, qrels, deps, *, gains=None)`：
    - 给了 `gains` 时，相关集合是等级 2 的块，并多算一个 `ndcg_graded@5`；
    - 排除原因除了 full_context，另有 `gold_unanswerable` 和 `no_gold_evidence`。
  - 工作流行新增两个字段：`answerability_gold`（三态或 null）、`answerable_gold`（bool 或 null）。由 CLI 每次运行时按 qid 从金标填入，所以续跑读回的旧行也会补上。
  - `build_report(..., labels: str = "silver", agreement: dict | None = None)`：labels 为 gold 时：
    - 换用金标的局限说明 `GOLD_CAVEATS`，必须写到四点：子 agent 预标、人逐题审核；等级 0 的判定主要来自预标；候选池深度是 10 加人工搜索；盲标一致率是人和子 agent 之间的，不是两位人工之间的（5.6 节要求的双人标注，这里只有一位人工）。
    - 可答性表多一行 B2"AmazonQA 自带分类器"：直接拿银标当预测，对比金标。
    - 新增三节：三态混淆矩阵、是非题结论准确率、一致率。一致率包括：金标与银标（二值 kappa）、预标与终审（三态 kappa，以及改动比例）、盲标题上盲标与预标（三态 kappa）。
    - 按评论量分段对比工作流和 B1：只在混合检索商品上，分 ≤20k、20k–60k、>60k tokens 三档。
  - CLI `--labels PATH`：
    - 题集是该 split 里所有已标注的题，忽略 `--sample`；
    - 检索评测只跑这些题；
    - `config.json` 记下 `labels_path` 和金标文件的 SHA-256；
    - `GoldError` 时退出码 2。

- [ ] **Step 1: 写失败的测试**

```python
def test_load_gold_rejects_other_dataset_and_missing_chunks(self):     # GoldError both ways
def test_gold_qrels_only_answerable_and_grade_two_is_relevant(self):
def test_cohen_kappa_known_values(self):   # perfect → 1.0; [a,a,b,b] vs [a,b,a,b] → 0.0; constant → None
def test_ndcg_graded(self):                # gains {x:2, y:1}, ranked [y, x] → (1/1 + 2/log2(3)) / (2/1 + 1/log2(3))
def test_three_state_confusion_excludes_service_failures(self):
def test_gold_report_has_b2_confusion_agreement_and_gold_caveats(self):
def test_cli_labels_mode_uses_labelled_questions_and_exits_2_on_mismatch(self):
```

- [ ] **Step 2:** 运行。Expected：FAIL。
- [ ] **Step 3:** 实现。银标模式的输出不变，5a 的测试要照样通过。
- [ ] **Step 4:** 跑全量测试，Expected：OK。
- [ ] **Step 5: 提交** `feat: evaluate against gold labels with agreement and three-state metrics`

### Task 6: 抽题与子 agent 预标（流程任务）

不调用 API key，但要加载 BGE-M3 和重排模型，并占用会话额度。

- [ ] **Step 1:** `python -m reviewproof.annotate pick --set gold_v1 --dev 100 --test 100`。Expected：200 题，其中盲标题 40 题；`annotations/gold_v1/items.jsonl` 写出。记下候选池大小的分布（均值和最大值）。
- [ ] **Step 2:** 按 qid 把题分成每批 10 题，写到 `annotations/gold_v1/batches/batch-NN.txt`。
- [ ] **Step 3: 派子 agent 预标。** 每批一个 Opus 子 agent，同时最多 4 个。给每个子 agent 的指令：
  - 先读 `docs/annotation_guidelines_cn.md`；
  - 只用 `packet` 和 `search` 两个子命令；
  - 不读 `data/`、不查数据库、不看 `reports/`；
  - 结果写到 `annotations/gold_v1/drafts/batch-NN.jsonl`，annotator 记为 `agent:claude-opus-5-5`。
- [ ] **Step 4:** 每批完成后跑 `import-drafts`。被拒的行退回给同一个子 agent 改，直到这一批全部导入。
- [ ] **Step 5:** `status`。Expected：200 题都是 drafted 或之后的状态。提交 `annotations/gold_v1/`（items、batches、drafts），提交信息 `data: add gold_v1 sample and agent drafts`。
- [ ] **Step 6: 停下来交给用户审核。** 告诉用户：启动命令 `REVIEWPROOF_ANNOTATION=1 python -m reviewproof.api`、页面地址、200 题中有 40 题是盲标题，预计每题 1–2 分钟。

### Task 7: 导出金标、正式评测、文档（等用户审核完再做）

- [ ] **Step 1:** `export --set gold_v1`。Expected：`labels.jsonl` 有 200 行。提交 `data: add gold_v1 labels`。
- [ ] **Step 2: 征得用户同意后**，跑：
  - `python -m reviewproof.eval --split dev --name gold-dev --labels annotations/gold_v1/labels.jsonl`
  - test 同样跑一遍，`--name gold-test`。

  两个都放后台。Expected：两份报告都包含 B2、混淆矩阵、一致率和分段对比。
- [ ] **Step 3: 文档。**
  - CLAUDE.md：加入 annotate 命令和"Phase 5b"一节。
  - README：更新进度，加上金标报告的链接。
  - 结论只写报告里的数字和区间。
- [ ] **Step 4: 提交** `docs: add phase 5b gold reports and docs`
