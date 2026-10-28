# 评测报告：gold-test

## 1. 概要

- name: `gold-test`
- split: `test`
- started_at: `2026-10-28T16:49:02+00:00`
- git_commit: `c65e6f9f71d67cede741ee4c6064a88e1517c779`
- dataset_id: `amazonqa-val-v1`
- source_sha256: `866095e6cce9abe5bc142e0ff1326737d57b5f24b19670123c6006d618bd941f`
- embedding_model: `BAAI/bge-m3`
- rerank_model: `BAAI/bge-reranker-v2-m3`
- llm_model: `claude-haiku-5-5`
- retrieval_params: `{'rrf_k': 60, 'channel_top_k': 50, 'rerank_pool': 40, 'evidence_k': 8, 'full_context_max_tokens': 6000}`
- max_rewrites: `1`
- seed: `20261027`
- questions_in_split: `100`
- sample: `100`
- systems: `['workflow', 'b0_no_evidence', 'b1_full_context']`
- labels: `gold`
- labels_path: `annotations/gold_v1/labels.jsonl`
- labels_sha256: `f22712d173066bdf87391cfeb8e567a7356c28c56c3bbc23dd366edf634f82ac`
- cost_note: `Claude Haiku 5.5 prices as of 2026-10-27; each call priced at its own tier (over 100k input tokens is long-context)`

- **金标**由子 agent 按 docs/annotation_guidelines_cn.md 预标，再由人工逐题审核（通过或修改）得到；每题都经过人确认。
- **证据等级 0（无关）**主要来自预标：审核时预标为无关的块默认折叠，人工不一定逐块复核，所以漏标的相关块会让召回偏低。
- **候选池**是四种检索方案各取前 10 的并集，加上标注时搜索找到的池外块；池外没被看到的相关块不计入。
- **盲标一致率**是人和子 agent 之间的，不是两位人工之间的；定位文档 5.6 节要求的双人标注，这里只有一位人工。
- 子 agent 与被测系统同属 Claude 模型家族，审核能纠正明显的错误，但同类偏差未必都能发现。

## 2. 检索

共 100 题，计入指标 62 题；排除：{'full_context': 30, 'gold_unanswerable': 8}；各方案出错次数：{'bm25': 0, 'dense': 0, 'hybrid': 0, 'hybrid_rerank': 0}。每格为 均值 [95% 置信区间]。

| 方案 | n | recall@5 | recall@10 | recall@20 | recall@40 | hit@5 | mrr@10 | ndcg@5 | ndcg_graded@5 |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 62 | 0.321 [0.244, 0.401] | 0.436 [0.355, 0.523] | 0.581 [0.502, 0.662] | 0.711 [0.633, 0.785] | 0.742 [0.629, 0.855] | 0.584 [0.481, 0.696] | 0.411 [0.336, 0.493] | 0.522 [0.457, 0.596] |
| dense | 62 | 0.368 [0.296, 0.447] | 0.536 [0.452, 0.612] | 0.688 [0.612, 0.760] | 0.804 [0.734, 0.871] | 0.790 [0.694, 0.887] | 0.674 [0.574, 0.775] | 0.513 [0.425, 0.601] | 0.606 [0.535, 0.679] |
| hybrid | 62 | 0.391 [0.309, 0.474] | 0.550 [0.464, 0.632] | 0.687 [0.605, 0.765] | 0.861 [0.792, 0.930] | 0.774 [0.661, 0.871] | 0.656 [0.552, 0.758] | 0.518 [0.436, 0.610] | 0.630 [0.560, 0.705] |
| hybrid_rerank | 62 | 0.497 [0.415, 0.580] | 0.659 [0.575, 0.738] | 0.784 [0.706, 0.858] | 0.861 [0.792, 0.930] | 0.855 [0.758, 0.935] | 0.752 [0.661, 0.838] | 0.650 [0.565, 0.733] | 0.732 [0.665, 0.796] |

每题平均有 6.3 个参考证据块，所以 Recall@5 的上限约为 0.80，不要把接近这个值的分数读成检索偏弱。

hybrid_rerank 减去其他方案（配对差值）：

| 对比 | hit@5 | ndcg@5 |
|---|---|---|
| hybrid_rerank − bm25 | 0.113 [0.000, 0.226] | 0.238 [0.148, 0.318] |
| hybrid_rerank − dense | 0.065 [-0.016, 0.161] | 0.136 [0.075, 0.199] |
| hybrid_rerank − hybrid | 0.081 [0.016, 0.161] | 0.132 [0.074, 0.192] |

按品类的 hit@5：

| 品类 | bm25 | dense | hybrid | hybrid_rerank |
|---|---|---|---|---|
| Automotive | 0.643 [0.357, 0.857] | 0.643 [0.429, 0.857] | 0.643 [0.357, 0.857] | 0.786 [0.571, 1.000] |
| Beauty | 0.824 [0.647, 1.000] | 0.824 [0.647, 1.000] | 0.765 [0.588, 0.941] | 0.824 [0.647, 1.000] |
| Electronics | 0.824 [0.647, 1.000] | 0.882 [0.706, 1.000] | 0.882 [0.706, 1.000] | 0.882 [0.706, 1.000] |
| Home_and_Kitchen | 0.643 [0.357, 0.857] | 0.786 [0.571, 1.000] | 0.786 [0.571, 1.000] | 0.929 [0.786, 1.000] |

## 3. 可答性

对照人工金标。"答了" = answered 或 partial；"拒答" = unanswerable 或不在服务范围。服务故障不进入分母，单独计数。

B2 AmazonQA 自带分类器：直接拿银标当作预测，和金标对比（定位文档第 6 节的 B2 基线）。

B1 只在混合检索商品的题上运行（全量上下文商品上它与工作流相同），所以和 B1 比较时请看"完整工作流（与 B1 同题）"一行，它只统计 B1 也跑了的题。

| 系统 | 计入/总数 | 服务故障 | 正确拒答率 | 误拒率 | 答复 F1 | 准确率 | 三态分布 | 服务故障被报成无法确认 |
|---|---|---|---|---|---|---|---|---|
| 完整工作流 | 100/100 | 0 | 0.850 [0.650, 1.000] | 0.175 [0.100, 0.263] | 0.886 [0.825, 0.936] | 0.830 [0.750, 0.900] | {'answered': 46, 'unanswerable': 31, 'partial': 23} | 0 |
| B0 不给证据 | 100/100 | 0 | 0.900 [0.750, 1.000] | 0.912 [0.850, 0.963] | 0.157 [0.052, 0.253] | 0.250 [0.170, 0.340] | {'unanswerable': 91, 'answered': 9} | 0 |
| B1 全量上下文 | 70/70 | 0 | 0.500 [0.125, 0.875] | 0.226 [0.113, 0.339] | 0.842 [0.761, 0.911] | 0.743 [0.629, 0.843] | {'answered': 38, 'partial': 14, 'unanswerable': 18} | 0 |
| 完整工作流（与 B1 同题） | 70/70 | 0 | 0.625 [0.250, 1.000] | 0.161 [0.081, 0.258] | 0.889 [0.818, 0.944] | 0.814 [0.714, 0.900] | {'answered': 39, 'partial': 16, 'unanswerable': 15} | 0 |
| B2 AmazonQA 自带分类器 | 100/100 | 0 | 0.550 [0.300, 0.750] | 0.163 [0.075, 0.250] | 0.859 [0.800, 0.912] | 0.780 [0.700, 0.860] | {'answered': 76, 'unanswerable': 24} | 0 |

## 3b. 三态混淆矩阵与是非题结论

行是金标，列是系统输出；服务故障不计入。三态一致率 = 系统三态与金标完全相同的比例。

**完整工作流**（n=100，三态一致率 0.720 [0.630, 0.810]）

| 金标 \ 系统 | answered | partial | unanswerable |
|---|---|---|---|
| answered | 42 | 8 | 5 |
| partial | 3 | 13 | 9 |
| unanswerable | 1 | 2 | 17 |

**B0 不给证据**（n=100，三态一致率 0.220 [0.140, 0.300]）

| 金标 \ 系统 | answered | partial | unanswerable |
|---|---|---|---|
| answered | 4 | 0 | 51 |
| partial | 3 | 0 | 22 |
| unanswerable | 2 | 0 | 18 |

**B1 全量上下文**（n=70，三态一致率 0.614 [0.500, 0.714]）

| 金标 \ 系统 | answered | partial | unanswerable |
|---|---|---|---|
| answered | 33 | 6 | 4 |
| partial | 3 | 6 | 10 |
| unanswerable | 2 | 2 | 4 |

**完整工作流（与 B1 同题）**（n=70，三态一致率 0.714 [0.600, 0.814]）

| 金标 \ 系统 | answered | partial | unanswerable |
|---|---|---|---|
| answered | 35 | 4 | 4 |
| partial | 3 | 10 | 6 |
| unanswerable | 1 | 2 | 5 |

是非题结论准确率（金标结论为 yes / no / mixed 的是非题；系统没给结论算错）：

| 系统 | n | 准确率 |
|---|---|---|
| 完整工作流 | 15 | 0.533 [0.267, 0.733] |
| B1 全量上下文 | 12 | 0.667 [0.417, 0.917] |
| 完整工作流（与 B1 同题） | 12 | 0.500 [0.167, 0.750] |

## 3c. 一致率

| 对比 | n | Cohen's kappa | 一致比例 | 备注 |
|---|---|---|---|---|
| 金标 vs AmazonQA 银标（二值） | 100 | 0.360 | 0.780 | 银标基于 10 个 BM25 片段 |
| 子 agent 预标 vs 人工终审（三态） | 100 | 0.983 | 0.990 | 终审改动了 0.050 的题（任一项不同即算） |
| 人工盲标 vs 子 agent 预标（三态） | 20 | 0.919 | 0.950 | 人和子 agent 之间，不是两位人工之间 |

## 3d. 按评论量分段（混合检索商品，工作流 vs B1）

只统计 B1 也跑了的题，按商品评论总 token 数分档；数值是三态一致率。用来判断全量上下文阈值该不该调。

| 评论量 | n | 完整工作流 | B1 全量上下文 |
|---|---|---|---|
| ≤20k | 38 | 0.658 [0.500, 0.789] | 0.605 [0.447, 0.763] |
| 20k–60k | 29 | 0.793 [0.655, 0.931] | 0.621 [0.448, 0.793] |
| >60k | 3 | 0.667 [0.000, 1.000] | 0.667 [0.000, 1.000] |

## 4. 成本与延迟

B1 只在混合检索商品的题上运行（全量上下文商品上它与工作流相同），所以和 B1 比较时请看"完整工作流（与 B1 同题）"一行，它只统计 B1 也跑了的题。

| 系统 | n | 总成本 (USD) | 每题成本 | P50 延迟 (s) | P95 延迟 (s) | 平均 LLM 调用 | 降级率 | 越界引用 |
|---|---|---|---|---|---|---|---|---|
| 完整工作流 | 100 | 0.1102 | 0.00110 | 14.2 | 26.9 | 3.1 | 0.000 | 0 |
| B0 不给证据 | 100 | 0.0096 | 0.00010 | 1.5 | 2.7 | 1.0 | 0.000 | 0 |
| B1 全量上下文 | 70 | 0.7720 | 0.01103 | 7.7 | 12.6 | 2.8 | 0.000 | 0 |
| 完整工作流（与 B1 同题） | 70 | 0.0770 | 0.00110 | 17.3 | 28.8 | 3.4 | 0.000 | 0 |

成本按 Claude Haiku 5.5 价格（2026-10-27）计算，每次调用按它自己的输入量选价格档（超过 10 万 token 为长上下文价格）。

## 5. 失败案例

以下为完整工作流的案例（每类最多 5 题），对照人工金标。

**误拒（金标可答，系统拒答）**

- `11754` · unanswerable · "Verdict: Can't confirm\nThe reviews don't confirm this."
- `39893` · unanswerable · "Verdict: Can't confirm\nThe reviews don't confirm this."
- `43266` · unanswerable · "Verdict: Can't confirm\nThe reviews don't confirm this."
- `44000` · unanswerable · "Verdict: Can't confirm\nThe reviews don't confirm this."
- `51664` · unanswerable · "Verdict: Can't confirm\nThe reviews don't confirm this."

**误答（金标不可答，系统答了）**

- `58884` · answered · "Reviewers describe the Avante's design as a slanted front (front-load) with the toast visible from the front, rather than a raised portion on the front. [1][2][3]\nOne reviewer says the front-load desi"
- `67013` · partial · 'Verdict: Yes\nReviewers describe the drive as attached to the key ring by a rubber loop, and say it does not bind up on the key ring and is light enough not to weigh down the ring. [1][2]\nOne reviewer '
- `76068` · partial · 'One reviewer found the Bluetooth module worked only for calls and the music cut out every two to three seconds, with the same result on a different phone; they returned it for a refund. [1]\nAnother re'

## 6. 局限与下一步

- 金标每个 split 约 100 题，区间仍然较宽；结论只写表中的数值和区间。
- 金标的局限见第 1 节；结论支持率（逐条核对引用）不在本报告里。
- 全量上下文商品的题不进入检索指标，只出现在可答性评测里。
