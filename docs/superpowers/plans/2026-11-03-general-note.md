# 通用参考（General Note）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the reviews can't answer a question (unanswerable), can answer only part of it (partial with listed gaps), or the question is out of scope, append a clearly labelled, uncited note: what is generally true of this kind of product, or where the shopper could check. Answerability, citations and the evaluated path stay unchanged.

**Architecture:** A new LangGraph node `advise` runs after `finalize`, only when a note applies. It makes one extra Claude call with a new schema `GeneralNote` and a prompt that holds the question, the product title and category, and the missing parts or out-of-scope reason, never review text. The note is a separate field (`QAResult.note`), never part of `answer_text`. `stream_answer` / `answer_question` take `advise: bool = False`; the API and the workflow CLI pass `True`, evaluation leaves it off.

**Tech Stack:** Python 3.10+, LangGraph, Anthropic SDK (`ClaudeLLM.structured`), Pydantic, FastAPI SSE, Vue 3 + TypeScript, unittest, Vitest.

**Spec:** agreed in chat on 2026-11-03 (no separate design doc). The decisions, verbatim from that conversation:
- Note on unanswerable, on partial (only for the parts the reviews don't cover), and on out-of-scope (price/stock/shipping/seller → where to check; medical/legal → consult a professional). No note on answered runs or on service failures.
- Two kinds: "general" (what is typical for this kind of product, not tied to its parameters) and "suggestion" (where to check, for product-specific questions). The note node decides the kind, so the plan step and the evaluated path don't change.
- The note prompt sees no review text. Product title and category may be given; the note must never state facts about this specific product or model, and must not recommend other brands or products.
- Shown under a fixed heading rendered by code, separate from the review findings, without citations. The answerability status stays as it is.
- A failed note call drops the note and is recorded in the trace; the run is not marked failed or degraded.
- The note is not evaluated; evaluation runs with it off.

## Global Constraints

- Questions, community answers, snippets and labels from the `questions` table never reach any prompt (the user's question text, as asked, is fine).
- Fixed copy exists only in Chinese and English; language is chosen by `is_chinese(question)`.
- Headings (exact copy):
  - zh general: `以下不是评论内容，是这类商品的一般情况，仅供参考：`
  - zh suggestion: `以下不是评论内容，是可以去哪里确认的建议：`
  - en general: `Not from the reviews: what is generally true of this kind of product, for reference only.`
  - en suggestion: `Not from the reviews: where you could check this.`
- `service_status` semantics don't change; the note never sets `service_failed` or `degraded`.
- `answer_text`, `status`, `verdict`, `claims`, `citations`, `unknowns` are identical with and without the note.
- Tests use `unittest`; database tests subclass `tests.support.DatabaseTestCase`.

## Review Focus

1. A "general" note that slips in specs of this exact model (prompt-only guard): checked by reading the notes in the Task 5 real run.
2. A product whose `title` is NULL: the prompt must still work and say the title is unknown (Task 1 test `test_prompt_has_title_category_and_no_reviews` covers both cases).
3. Note text containing `[1]`: the frontend must show it as plain text, not a citation button (Task 3 test).
4. The note leaking into `answer_text` or changing the badge: Task 1 asserts `answer_text` is unchanged; Task 3 asserts the badge still reads from `status`.
5. Evaluation silently starting to call the note (cost and `llm_calls` would drift from the reports): Task 2 test on the eval call.

---

### Task 1: `advise` node, schema, prompt and copy

**Files:**
- Modify: `reviewproof/workflow/schemas.py` (add `GeneralNote`)
- Modify: `reviewproof/workflow/prompts.py` (add `ADVISE_SYSTEM`, `advise_prompt`; update module docstring)
- Modify: `reviewproof/workflow/answer.py` (add `Note`, `render_note`, four COPY entries)
- Modify: `reviewproof/workflow/state.py` (`QAState.note`; `TraceEntry.node` comment lists `advise`)
- Modify: `reviewproof/workflow/nodes.py` (`WorkflowNodes(..., advise: bool)`, `advise` node, `after_finalize`)
- Modify: `reviewproof/workflow/graph.py` (wire the node; `advise` parameter; `QAResult.note`)
- Test: `tests/test_workflow_note.py` (new), `tests/test_workflow_answer.py`

**Interfaces:**
- Produces:
  - `class GeneralNote(BaseModel): kind: Literal["general", "suggestion"]; text: str`
  - `@dataclass(frozen=True) class Note: kind: str; heading: str; text: str` in `answer.py`
  - `render_note(question: str, kind: str, text: str) -> Note`
  - `advise_prompt(question: str, *, title: str | None, category: str, missing: Sequence[str], out_of_scope_reason: str) -> str`
  - `QAResult.note: Note | None = None` (last field)
  - `stream_answer(..., advise: bool = False)`, `answer_question(..., advise: bool = False)`
  - trace node name `"advise"`; statuses `ok` / `failed`

- [ ] **Step 1: Write the failing tests**

In `tests/test_workflow_answer.py`:

```python
def test_render_note_headings(self):
    self.assertEqual(render_note("音量可以调吗？", "general", "一般可以。"),
                     Note("general", "以下不是评论内容，是这类商品的一般情况，仅供参考：", "一般可以。"))
    self.assertEqual(render_note("Is it loud?", "suggestion", "Ask the seller.").heading,
                     "Not from the reviews: where you could check this.")
```

New `tests/test_workflow_note.py`, a `DatabaseTestCase` reusing the fixtures of `tests/test_workflow.py` (products A, B, C; deps with `full_context_max_tokens=0`), with `ask(script, ..., advise=True)` that passes `advise` through to `answer_question`. `note(kind, text)` builds `GeneralNote`.

```python
def test_unanswerable_run_gets_a_note(self):
    r = self.ask({"QueryPlan": [plan("loud")], "EvidenceAssessment": [assess("unanswerable")],
                  "GeneralNote": [note("suggestion", "Check the product page.")]})
    self.assertEqual(r.status, "unanswerable")
    self.assertEqual(r.answer_text, "Verdict: Can't confirm\nThe reviews don't confirm this.")
    self.assertEqual(r.note, Note("suggestion", "Not from the reviews: where you could check this.",
                                  "Check the product page."))
    self.assertEqual(r.trace[-1].node, "advise")
    self.assertEqual(r.llm_calls, 3)

def test_partial_note_prompt_lists_only_the_missing_parts(self):
    # answered-style claim on A, assessment partial with missing=["battery life"]
    ...
    prompt = self.calls("GeneralNote")[0][2]
    self.assertIn("battery life", prompt)
    self.assertEqual(r.status, "partial")
    self.assertIsNotNone(r.note)

def test_partial_without_missing_parts_gets_no_note(self): ...   # missing=[] → no GeneralNote call, note None
def test_answered_run_gets_no_note(self): ...                     # no GeneralNote call, note None
def test_out_of_scope_run_gets_a_note_with_the_reason(self):
    # plan(route="out_of_scope", reason="price is not in reviews"); prompt contains the reason; note set
def test_service_failure_gets_no_note(self):
    # EvidenceAssessment raises LLMError → service_status "failed", no GeneralNote call, note None
def test_note_failure_keeps_the_answer(self):
    # GeneralNote raises LLMError → note None, service_status "ok", answer_text as without note,
    # trace[-1] == ("advise", "failed", ...)
def test_blank_note_text_is_dropped(self):
    # GeneralNote text "  " → note None, trace[-1].status "ok"
def test_note_is_off_by_default(self):
    # answer_question without advise on an unanswerable run → no GeneralNote call, note None
def test_prompt_has_title_category_and_no_reviews(self):
    # UPDATE products SET title = 'Acme Fan' WHERE product_id = 'A'; commit
    # prompt contains "Acme Fan" and "Electronics"; contains no review text of A ("Too loud" etc. —
    # every chunk text of A is absent); with title NULL (product B) the prompt contains "title unknown"
def test_unanswerable_answer_is_the_same_with_and_without_note(self):
    # same script twice, advise False/True: status, verdict, answer_text, citations equal
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `REVIEWPROOF_TEST_DATABASE_URL=postgresql://reviewproof:reviewproof@localhost:54329/reviewproof_test python -m unittest tests.test_workflow_note tests.test_workflow_answer -b`
Expected: FAIL / ERROR on `GeneralNote`, `Note`, `render_note` imports.

- [ ] **Step 3: Implement**

- `schemas.py`: `GeneralNote` as in Interfaces.
- `answer.py`: `Note`; COPY keys `note_general` / `note_suggestion` with the exact headings above; `render_note` picks language with `is_chinese(question)` and strips `text`.
- `prompts.py`: module docstring now says the note prompt also gets the product's title and category (from Amazon metadata in `products`) and no review text. `advise_prompt` lines, joined by blank lines: `Question:\n{question}`, `_language(question)`, `Product: {title or "title unknown"} (category: {category})`, then `The reviews did not cover: a; b` when `missing`, or `Outside what reviews can answer: {reason}` when `out_of_scope_reason`. `ADVISE_SYSTEM` (exact):

```text
A pre-purchase assistant answers questions about one product from its customer reviews. The reviews \
could not answer this question, or the parts listed. Write a short note that still helps the shopper; \
it is shown under a heading saying it is not from the reviews.

kind:
- "general": the question (or the uncovered part) asks what is typical for this kind of product, \
independent of this product's own model or specifications. Answer it from general knowledge, \
phrased as what is usual for this kind of product.
- "suggestion": the answer depends on this specific product (its model, size, specifications, \
compatibility, warranty) or on the seller (price, stock, shipping). Say where the shopper could \
check: the specifications on the product page, the manufacturer's manual or website, asking the \
seller, or measuring their own setup.

Rules:
- Never state facts about this specific product or model, even ones you believe you know.
- Do not recommend other brands or products.
- For medical or legal questions, only suggest asking a qualified professional.
- At most three sentences, in the answer language, with no citations or bracketed ids.
```

- `nodes.py`:
  - `WorkflowNodes.__init__(self, conn, llm, retrieval, max_rewrites, advise: bool = False)`.
  - `after_finalize(state) -> str`: `"advise"` when `self.advise` and not `service_failed` and (`route == "out_of_scope"` or `status == "unanswerable"` or (`status == "partial"` and `unknowns`)), else `END`.
  - `advise(state) -> dict`: read `title, category` from `products` (then `conn.rollback()`); `missing = unknowns` when partial, else `()`; reason only when out of scope; call `self.llm.structured(system=ADVISE_SYSTEM, prompt=advise_prompt(...), schema=GeneralNote)`. On `LLMError` or `psycopg.Error` (roll back): no `note`, trace `("advise", "failed", ...)`. Blank text: no `note`, trace `ok` with detail `empty note dropped`. Otherwise `note = render_note(...)`, `usage` appended, trace `ok` with `kind=...`. Never touches `service_failed` / `degraded`.
- `graph.py`: node list gains `"advise"`; replace `finalize → END` with `add_conditional_edges("finalize", nodes.after_finalize, ["advise", END])` and `advise → END`; module docstring shows `→ finalize → [advise]`; `stream_answer` / `answer_question` take and pass `advise`; `QAResult(note=state.get("note"))`.

- [ ] **Step 4: Run the tests to see them pass, then the whole suite**

Run: same command as Step 2, then `REVIEWPROOF_TEST_DATABASE_URL=... python -m unittest discover -s tests -b`
Expected: all pass (existing tests untouched: the default is off).

- [ ] **Step 5: Commit**

```bash
git add reviewproof/workflow tests/test_workflow_note.py tests/test_workflow_answer.py
git commit -m "feat: labelled general note after refusals, partial gaps and out-of-scope questions"
```

### Task 2: API, workflow CLI and evaluation

**Files:**
- Modify: `reviewproof/api/qa.py` (pass `advise=True`; `done` gains `"note"`)
- Modify: `reviewproof/workflow/cli.py` (pass `advise=True`; `format_result` prints the note)
- Modify: `reviewproof/eval/workflow_eval.py` (pass `advise=False` explicitly, with a comment that the note is not evaluated)
- Test: `tests/test_api_qa.py`, `tests/test_workflow_cli.py`, `tests/test_eval_workflow.py`

**Interfaces:**
- Consumes: `stream_answer(..., advise=...)`, `answer_question(..., advise=...)`, `QAResult.note: Note | None`
- Produces: SSE `done` field `note: {"kind": str, "heading": str, "text": str} | null` (`asdict(note)` or `None`)

- [ ] **Step 1: Write the failing tests**

- `test_api_qa.py`: `test_unanswerable_run_streams_the_note` — script with assessment `unanswerable` and `GeneralNote: [GeneralNote(kind="general", text="Usually adjustable.")]`; the `done` event has `note == {"kind": "general", "heading": "Not from the reviews: what is generally true of this kind of product, for reference only.", "text": "Usually adjustable."}`; no `answer_chunk` contains "Usually adjustable."; the saved run (`GET /v1/runs/{id}`) has the same `note`. `test_answered_run_has_no_note` — existing answered script, `done["note"] is None`.
- `test_workflow_cli.py`: `test_text_output_shows_the_note` — unanswerable script with a note; stdout contains the heading line followed by the note text, after the answer and before `status=`.
- `test_eval_workflow.py`: `test_workflow_rows_never_call_the_note` — an unanswerable workflow row with a `FakeLLM` script that has no `GeneralNote`; the row has no `error` and the LLM calls contain no `GeneralNote`.

- [ ] **Step 2: Run them to see them fail**

Run: `REVIEWPROOF_TEST_DATABASE_URL=... python -m unittest tests.test_api_qa tests.test_workflow_cli tests.test_eval_workflow -b`
Expected: the API and CLI tests fail (no `note` key / no heading); the eval test passes already (default off) — keep it as the guard.

- [ ] **Step 3: Implement** as listed under Files. In `format_result`, when `r.note`: a blank line, `r.note.heading`, `r.note.text`, inserted right after `r.answer_text`.

- [ ] **Step 4: Run them, then the whole suite**

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add reviewproof/api/qa.py reviewproof/workflow/cli.py reviewproof/eval/workflow_eval.py tests/test_api_qa.py tests/test_workflow_cli.py tests/test_eval_workflow.py
git commit -m "feat: stream the general note from the API and CLI; evaluation keeps it off"
```

### Task 3: Frontend

**Files:**
- Modify: `frontend/src/api/types.ts` (`GeneralNote` interface; `DoneEvent.note: GeneralNote | null`)
- Modify: `frontend/src/components/AnswerPanel.vue` (note block)
- Modify: `frontend/src/components/ProgressList.vue` (label `advise: "补充参考"`)
- Modify: `frontend/src/style.css` (`.note` style: muted background, smaller text, distinct from `.answer-text`)
- Test: `frontend/src/components/AnswerPanel.test.ts`, plus any test fixture building a `DoneEvent` (add `note: null`)

**Interfaces:**
- Consumes: `done.note` from Task 2.
- Produces: `<aside class="note" data-test="note">` with `<p data-test="note-heading">` and `<p data-test="note-text">`.

- [ ] **Step 1: Write the failing tests** in `AnswerPanel.test.ts`:
  - `renders the note below the answer` — `done.note = {kind: "suggestion", heading: "以下不是评论内容，是可以去哪里确认的建议：", text: "可以问卖家。"}`; `[data-test=note-heading]` and `[data-test=note-text]` show them; the note element comes after `[data-test=answer]`.
  - `shows no note when there is none` — `note: null` → no `[data-test=note]`.
  - `note text is plain, not citation buttons` — text `"见说明书 [1]"` → no `[data-test=cite-ref]` inside the note.
  - `badge still follows the status` — unanswerable + note → badge label as before.

- [ ] **Step 2: Run to see them fail**

Run: `cd frontend && npm run typecheck && npm test`
Expected: typecheck fails on `note`, or the new tests fail.

- [ ] **Step 3: Implement** as listed under Files.

- [ ] **Step 4: Run to see them pass, and build**

Run: `cd frontend && npm run typecheck && npm test && npm run build`
Expected: all pass; `frontend/dist` rebuilt.

- [ ] **Step 5: Commit**

```bash
git add frontend/src
git commit -m "feat: show the general note apart from the review answer"
```

### Task 4: Docs

**Files:**
- Modify: `README.md` — feature list: refusals come with a labelled general note; evaluation numbers don't include it.
- Modify: `docs/product_positioning_cn.md` — the 通用大模型会"想当然" row (line 22) and the one-liner (section 9): reviews remain the only evidence; refusals add a clearly labelled, uncited general note.
- Modify: `docs/design_and_lessons_cn.md` — a new section on the note (why it was added on 2026-11-03, the two kinds, why it is a separate node that sees no reviews, why it runs after finalize, why it is not evaluated, its failure handling) and a row in the module design review table (choice / alternatives: fold into generation, replace refusals / evidence type: design argument, not evaluated / weakness: only a prompt keeps product-specific facts out).
- Modify: `docs/technical_report_cn.md` — section 3 (system): one paragraph on the note; section 4 or 5: evaluation runs with it off, so every number is unaffected.
- Modify: `docs/project_overview_cn.md` — architecture and feature lines.
- Local only (git-ignored): `CLAUDE.md` Phase 3 and Phase 4 paragraphs; then `cp CLAUDE.md .git/CLAUDE.md.local-backup`.

- [ ] **Step 1: Edit the files above.** Keep claims conservative: the note is not evaluated, and "not about this specific product" is enforced only by the prompt.
- [ ] **Step 2: Check** `git diff --stat` lists only the five tracked docs; `git status --short` doesn't list `CLAUDE.md`.
- [ ] **Step 3: Commit**

```bash
git add README.md docs/product_positioning_cn.md docs/design_and_lessons_cn.md docs/technical_report_cn.md docs/project_overview_cn.md
git commit -m "docs: general note after refusals"
```

### Task 5: Real run (needs the user's consent for Claude calls)

- [ ] **Step 1:** Ask the user before calling Claude. Then, with the API running, ask 4 questions through `POST /v1/qa` or the workflow CLI: one general unanswerable question, one product-specific unanswerable question, one partial question, one out-of-scope question (price). Use English and Chinese at least once each.
- [ ] **Step 2:** Read each note against Review Focus 1: no specs of this exact model, no other brands, kind fits the question. Record the run ids and notes in the final message; anything wrong goes back to `ADVISE_SYSTEM` with a new test where possible.
- [ ] **Step 3:** Screenshot the answer panel with a note.
