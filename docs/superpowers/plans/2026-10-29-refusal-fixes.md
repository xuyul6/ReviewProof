# 金标误拒修复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 减少金标评测里的误拒（金标可答、系统判不可答），同时不增加误答。

**Architecture:** 只改工作流的三处规则：核验时是非结论 `unknown` 的处理、证据评估提示词里三态的定义、规划提示词里的范围规则。检索、接口、前端不动。

**Tech Stack:** Python、LangGraph 工作流、Claude Haiku 5.5 结构化输出；测试用 `unittest` 和 `FakeLLM`。

**Spec:** `docs/annotation_guidelines_cn.md`（金标规范 gold-v1，第 2 节三态定义）；误拒分析见下方"背景"。

## 背景：25 道误拒的分析

金标 dev 11 道、test 14 道误拒（金标 partial 17、answered 8）：

1. **11 道：是非结论 `unknown` 把整题降成不可答。** 评估已判能答或部分能答，生成步骤给了 `unknown`，`final_status` 按 `71b7341` 的规则降为 unanswerable。dev 5 道、test 6 道。
2. **11 道：评估直接判不可答。** 多数只有"相近情形"的证据（别的年款、别的用法），规范第 2 节算 partial；还有直接回答（dev 27383）和意见分歧（规范算 answered）被判不可答。dev 5 道、test 6 道。
3. **3 道：范围规则拦掉。** dev 73170 问"这个包选哪个尺寸"，被当成"推荐别的商品"；test 两道问替换零件怎么买。

误拒集中在标注时就被标成"有争议"的题（25 道里 22 道，全部可答题里 52%）。

## Global Constraints

- **只按 dev 和规范改。** 原因 1、2 在 dev 上都有；原因 3 只改 dev 上出现的"选尺寸被当成推荐"，**不**为 test 上的"替换零件"加规则。分析时已经看过 test 的错例，所以修复后 test 的数字要注明"test 的错例分析参与过修复方案的讨论"。
- 不改金标、规范和评测代码。
- 提示词里仍然只有问题和这件商品的评论块。
- 真实 Claude 调用要先征得用户同意：dev 只跑工作流约 $0.11，test 约 $0.11。
- 服务故障仍然不能被说成"评论无法确认"。

## Review Focus

1. 是非题 `unknown` 且所有引用都被核验删掉时，必须仍是 unanswerable（不能出现"部分能回答"但没有任何引用）。
2. 非是非题不受规则 1 影响（`verdict=None`）。
3. 评估判 unanswerable 时仍然跳过生成，状态不会被抬高。
4. 规划失败退回规则时行为不变。
5. 修复后误答（金标不可答、系统答了）不能明显增加：dev 原来 4 道，复跑后逐题列出新增的误答。

---

### Task 1: 是非结论 `unknown` 最多降到 partial

**Files:**
- Modify: `reviewproof/workflow/answer.py`（`final_status`）、`reviewproof/workflow/prompts.py`（`GENERATE_SYSTEM` 最后一句）
- Test: `tests/test_workflow_answer.py`、`tests/test_workflow.py`

**Interfaces:**
- Produces: `final_status(assessed: str, claims: Sequence[AnswerClaim], verdict: str | None = None) -> str`，签名不变；规则改为：`assessed == "unanswerable"` 或没有留下的结论 → `"unanswerable"`；否则 `verdict == "unknown"` → `"partial"`；否则 `assessed`。

- [ ] **Step 1: 改测试**
  - `FinalStatusTests.test_rules`：`final_status("partial", one, verdict="unknown") == "partial"`；`final_status("answered", one, verdict="unknown") == "partial"`；`final_status("answered", [], verdict="unknown") == "unanswerable"`；`final_status("unanswerable", one, verdict="unknown") == "unanswerable"`。
  - `tests/test_workflow.py` 的 `test_yes_no_verdict_unknown_makes_the_run_unanswerable` 改名为 `test_yes_no_verdict_unknown_caps_the_run_at_partial`：评估 answered、生成 `verdict="unknown"` 且有一条有效引用 → `(status, verdict) == ("partial", "unknown")`，引用保留，答案文本里有"Can't confirm"和那条结论。
  - 新增 `test_yes_no_verdict_unknown_without_valid_citations_is_unanswerable`：同上但引用的块不在候选里 → `("unanswerable", "unknown")`，没有引用。
- [ ] **Step 2: 跑测试，确认失败**：`.venv/bin/python -m unittest tests.test_workflow_answer tests.test_workflow -b`。Expected：上面三处断言失败。
- [ ] **Step 3: 改 `final_status`**；`GENERATE_SYSTEM` 最后一句改为：`unknown` 时"是或否"报告为无法确认，结论作为相关证据照常显示。
- [ ] **Step 4: 跑测试，确认通过**，再跑全部测试。
- [ ] **Step 5: 提交** `fix: let an unknown yes/no verdict keep related evidence as partial`

### Task 2: 评估按规范判三态

**Files:**
- Modify: `reviewproof/workflow/prompts.py`（`ASSESS_SYSTEM`）
- Test: `tests/test_workflow_answer.py`

- [ ] **Step 1: 写测试** `test_assess_prompt_follows_the_gold_guideline`：`ASSESS_SYSTEM` 包含 "close case"（相近情形算 partial）、"disagree"（意见分歧算 answered）、"one reviewer"（一位买家的明确亲身陈述就够）。
- [ ] **Step 2: 跑测试，确认失败。**
- [ ] **Step 3: 改 `ASSESS_SYSTEM` 的三态定义**，对齐规范第 2 节：
  - answered：评论回答了问题的核心；一位买家明确的亲身陈述就够；买家意见不一也算 answered。
  - partial：只回答了其中几部分，或者只有相近情形（别的年款或尺寸、相似的用法）；没覆盖的写进 missing。
  - unanswerable：评论里没有任何和问题相关的内容，连相近情形也没有。
  - 保留"不用常识补"。
- [ ] **Step 4: 跑全部测试。**
- [ ] **Step 5: 提交** `fix: assess close cases as partial and split opinions as answered, per the gold guideline`

### Task 3: 选这件商品的尺寸或款式不算"推荐别的商品"

**Files:**
- Modify: `reviewproof/workflow/prompts.py`（`PLAN_SYSTEM`）
- Test: `tests/test_workflow_answer.py`

- [ ] **Step 1: 写测试** `test_choosing_a_size_of_this_product_is_a_review_question`：`PLAN_SYSTEM` 按 `"review_question"` 切开后，后半段包含 "size or variant"；前半段的推荐规则写成 "a different product"。
- [ ] **Step 2: 跑测试，确认失败。**
- [ ] **Step 3: 改 `PLAN_SYSTEM`**：超范围里"推荐别的商品"改为"推荐另一件商品来代替这件"；review_question 的列举里加上"这件商品选哪个尺寸或款式适合买家自己的东西"。
- [ ] **Step 4: 跑全部测试。**
- [ ] **Step 5: 提交** `fix: route choosing a size of this product to the reviews`

### Task 4: dev 复跑和逐题对比（征得同意后）

- [ ] **Step 1:** `python -m reviewproof.eval --split dev --name gold-dev-r2 --labels annotations/gold_v1/labels.jsonl --systems workflow`。Expected：100 题，0 服务故障。
- [ ] **Step 2:** 逐题对比 `gold-dev` 和 `gold-dev-r2` 的工作流结果：修好的误拒、新增的误拒、新增的误答、三态变化。新增误答逐题看金标理由。
- [ ] **Step 3:** 如果新增误答多于修好的误拒，停下来报告，不跑 test。

### Task 5: test 复跑一次、文档

- [ ] **Step 1:** `python -m reviewproof.eval --split test --name gold-test-r2 --labels annotations/gold_v1/labels.jsonl --systems workflow`。
- [ ] **Step 2: 文档。** README 和 `docs/design_and_lessons_cn.md` 加上修复前后的数字（只写报告里的数字和区间，注明 test 的错例参与过分析）；复盘文档加一节"金标之后的误拒修复"；本地 CLAUDE.md 更新 `unknown` 的规则。
- [ ] **Step 3: 提交** `docs: add refusal-fix reports and before/after numbers`

## 执行结果（2026-10-29）：没有成立，全部回滚

- Task 1–3 完成后 dev 复跑（`reports/eval/gold-dev-r2`）：准确率 0.850 [0.770, 0.920]，和修复前相同；误拒 11 → 4，误答 4 → 11。
- 回滚 Task 1 后再跑 dev（`reports/eval/gold-dev-r3`）：准确率仍是 0.850 [0.770, 0.920]；误拒 8，误答 7。
- 按用户决定停下：Task 1–3 全部回滚（`e022560`、`9535003`），工作流代码和 gold-test 测过的版本相同；Task 5 的 test 复跑没有做。
- 原因和教训见 `docs/design_and_lessons_cn.md` 9.5 节。
