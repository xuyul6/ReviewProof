# 评测报告：gold-dev

## 1. 概要

- name: `gold-dev`
- split: `dev`
- started_at: `2026-10-28T16:06:12+00:00`
- git_commit: `af1f531e6bb4fce7e63d12eda89b67c35f324ae9`
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

共 100 题，计入指标 56 题；排除：{'gold_unanswerable': 14, 'full_context': 30}；各方案出错次数：{'bm25': 0, 'dense': 0, 'hybrid': 0, 'hybrid_rerank': 0}。每格为 均值 [95% 置信区间]。

| 方案 | n | recall@5 | recall@10 | recall@20 | recall@40 | hit@5 | mrr@10 | ndcg@5 | ndcg_graded@5 |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 56 | 0.278 [0.207, 0.356] | 0.403 [0.315, 0.488] | 0.565 [0.471, 0.653] | 0.735 [0.649, 0.808] | 0.679 [0.554, 0.804] | 0.597 [0.480, 0.724] | 0.431 [0.337, 0.526] | 0.574 [0.491, 0.654] |
| dense | 56 | 0.354 [0.280, 0.433] | 0.490 [0.407, 0.577] | 0.690 [0.612, 0.762] | 0.810 [0.741, 0.874] | 0.821 [0.714, 0.911] | 0.586 [0.483, 0.687] | 0.477 [0.394, 0.567] | 0.612 [0.535, 0.690] |
| hybrid | 56 | 0.347 [0.272, 0.425] | 0.518 [0.432, 0.599] | 0.712 [0.628, 0.787] | 0.915 [0.861, 0.962] | 0.786 [0.679, 0.893] | 0.649 [0.540, 0.754] | 0.505 [0.405, 0.601] | 0.643 [0.559, 0.720] |
| hybrid_rerank | 56 | 0.494 [0.414, 0.583] | 0.684 [0.608, 0.760] | 0.807 [0.738, 0.875] | 0.915 [0.861, 0.962] | 0.893 [0.804, 0.964] | 0.739 [0.648, 0.832] | 0.642 [0.555, 0.729] | 0.758 [0.689, 0.822] |

每题平均有 6.6 个参考证据块，所以 Recall@5 的上限约为 0.76，不要把接近这个值的分数读成检索偏弱。

hybrid_rerank 减去其他方案（配对差值）：

| 对比 | hit@5 | ndcg@5 |
|---|---|---|
| hybrid_rerank − bm25 | 0.214 [0.125, 0.321] | 0.211 [0.135, 0.286] |
| hybrid_rerank − dense | 0.071 [0.000, 0.161] | 0.166 [0.085, 0.243] |
| hybrid_rerank − hybrid | 0.107 [0.018, 0.214] | 0.138 [0.068, 0.212] |

按品类的 hit@5：

| 品类 | bm25 | dense | hybrid | hybrid_rerank |
|---|---|---|---|---|
| Automotive | 0.800 [0.500, 1.000] | 0.900 [0.700, 1.000] | 0.900 [0.700, 1.000] | 0.900 [0.700, 1.000] |
| Beauty | 0.769 [0.538, 1.000] | 0.846 [0.615, 1.000] | 0.692 [0.462, 0.923] | 0.923 [0.769, 1.000] |
| Electronics | 0.647 [0.412, 0.882] | 0.824 [0.647, 1.000] | 0.824 [0.647, 1.000] | 0.941 [0.824, 1.000] |
| Home_and_Kitchen | 0.562 [0.312, 0.812] | 0.750 [0.500, 0.938] | 0.750 [0.500, 0.938] | 0.812 [0.562, 1.000] |

## 3. 可答性

对照人工金标。"答了" = answered 或 partial；"拒答" = unanswerable 或不在服务范围。服务故障不进入分母，单独计数。

B2 AmazonQA 自带分类器：直接拿银标当作预测，和金标对比（定位文档第 6 节的 B2 基线）。

B1 只在混合检索商品的题上运行（全量上下文商品上它与工作流相同），所以和 B1 比较时请看"完整工作流（与 B1 同题）"一行，它只统计 B1 也跑了的题。

| 系统 | 计入/总数 | 服务故障 | 正确拒答率 | 误拒率 | 答复 F1 | 准确率 | 三态分布 | 服务故障被报成无法确认 |
|---|---|---|---|---|---|---|---|---|
| 完整工作流 | 100/100 | 0 | 0.826 [0.652, 0.957] | 0.143 [0.065, 0.221] | 0.898 [0.840, 0.944] | 0.850 [0.780, 0.910] | {'partial': 25, 'unanswerable': 30, 'answered': 45} | 0 |
| B0 不给证据 | 100/100 | 0 | 0.913 [0.783, 1.000] | 0.883 [0.805, 0.948] | 0.205 [0.095, 0.326] | 0.300 [0.210, 0.400] | {'unanswerable': 89, 'answered': 11} | 0 |
| B1 全量上下文 | 70/70 | 0 | 0.929 [0.786, 1.000] | 0.071 [0.018, 0.143] | 0.954 [0.909, 0.991] | 0.929 [0.871, 0.986] | {'partial': 17, 'answered': 36, 'unanswerable': 17} | 0 |
| 完整工作流（与 B1 同题） | 70/70 | 0 | 0.786 [0.500, 1.000] | 0.143 [0.054, 0.250] | 0.897 [0.826, 0.950] | 0.843 [0.743, 0.929] | {'partial': 17, 'unanswerable': 19, 'answered': 34} | 0 |
| B2 AmazonQA 自带分类器 | 100/100 | 0 | 0.565 [0.348, 0.783] | 0.156 [0.078, 0.234] | 0.855 [0.791, 0.910] | 0.780 [0.700, 0.860] | {'answered': 75, 'unanswerable': 25} | 0 |

## 3b. 三态混淆矩阵与是非题结论

行是金标，列是系统输出；服务故障不计入。三态一致率 = 系统三态与金标完全相同的比例。

**完整工作流**（n=100，三态一致率 0.710 [0.620, 0.790]）

| 金标 \ 系统 | answered | partial | unanswerable |
|---|---|---|---|
| answered | 41 | 10 | 3 |
| partial | 4 | 11 | 8 |
| unanswerable | 0 | 4 | 19 |

**B0 不给证据**（n=100，三态一致率 0.250 [0.170, 0.340]）

| 金标 \ 系统 | answered | partial | unanswerable |
|---|---|---|---|
| answered | 4 | 0 | 50 |
| partial | 5 | 0 | 18 |
| unanswerable | 2 | 0 | 21 |

**B1 全量上下文**（n=70，三态一致率 0.714 [0.600, 0.814]）

| 金标 \ 系统 | answered | partial | unanswerable |
|---|---|---|---|
| answered | 29 | 9 | 0 |
| partial | 6 | 8 | 4 |
| unanswerable | 1 | 0 | 13 |

**完整工作流（与 B1 同题）**（n=70，三态一致率 0.714 [0.600, 0.814]）

| 金标 \ 系统 | answered | partial | unanswerable |
|---|---|---|---|
| answered | 30 | 5 | 3 |
| partial | 4 | 9 | 5 |
| unanswerable | 0 | 3 | 11 |

是非题结论准确率（金标结论为 yes / no / mixed 的是非题；系统没给结论算错）：

| 系统 | n | 准确率 |
|---|---|---|
| 完整工作流 | 12 | 0.583 [0.333, 0.833] |
| B1 全量上下文 | 9 | 0.778 [0.556, 1.000] |
| 完整工作流（与 B1 同题） | 9 | 0.667 [0.333, 1.000] |

## 3c. 一致率

| 对比 | n | Cohen's kappa | 一致比例 | 备注 |
|---|---|---|---|---|
| 金标 vs AmazonQA 银标（二值） | 100 | 0.397 | 0.780 | 银标基于 10 个 BM25 片段 |
| 子 agent 预标 vs 人工终审（三态） | 100 | 1.000 | 1.000 | 终审改动了 0.030 的题（任一项不同即算） |
| 人工盲标 vs 子 agent 预标（三态） | 20 | 0.593 | 0.750 | 人和子 agent 之间，不是两位人工之间 |

## 3d. 按评论量分段（混合检索商品，工作流 vs B1）

只统计 B1 也跑了的题，按商品评论总 token 数分档；数值是三态一致率。用来判断全量上下文阈值该不该调。

| 评论量 | n | 完整工作流 | B1 全量上下文 |
|---|---|---|---|
| ≤20k | 38 | 0.737 [0.579, 0.868] | 0.763 [0.632, 0.895] |
| 20k–60k | 23 | 0.609 [0.391, 0.826] | 0.565 [0.348, 0.783] |
| >60k | 9 | 0.889 [0.667, 1.000] | 0.889 [0.667, 1.000] |

## 4. 成本与延迟

B1 只在混合检索商品的题上运行（全量上下文商品上它与工作流相同），所以和 B1 比较时请看"完整工作流（与 B1 同题）"一行，它只统计 B1 也跑了的题。

| 系统 | n | 总成本 (USD) | 每题成本 | P50 延迟 (s) | P95 延迟 (s) | 平均 LLM 调用 | 降级率 | 越界引用 |
|---|---|---|---|---|---|---|---|---|
| 完整工作流 | 100 | 0.1121 | 0.00112 | 14.5 | 27.1 | 3.1 | 0.000 | 0 |
| B0 不给证据 | 100 | 0.0094 | 0.00009 | 1.6 | 2.6 | 1.0 | 0.000 | 0 |
| B1 全量上下文 | 70 | 1.9983 | 0.02855 | 7.1 | 14.2 | 2.7 | 0.000 | 0 |
| 完整工作流（与 B1 同题） | 70 | 0.0733 | 0.00105 | 17.9 | 27.5 | 3.3 | 0.000 | 0 |

成本按 Claude Haiku 5.5 价格（2026-10-27）计算，每次调用按它自己的输入量选价格档（超过 10 万 token 为长上下文价格）。

## 5. 失败案例

以下为完整工作流的案例（每类最多 5 题），对照人工金标。

**误拒（金标可答，系统拒答）**

- `10844` · unanswerable · "The reviews don't confirm this."
- `1505` · unanswerable · "Verdict: Can't confirm\nThe reviews don't confirm this."
- `23528` · unanswerable · "Verdict: Can't confirm\nThe reviews don't confirm this."
- `23674` · unanswerable · "Verdict: Can't confirm\nThe reviews don't confirm this."
- `27383` · unanswerable · "Verdict: Can't confirm\nThe reviews don't confirm this."

**误答（金标不可答，系统答了）**

- `12206` · partial · 'Verdict: Reviews disagree\nOne reviewer says the lamp did not fit their JVC rear projection TV. [1]\nOne reviewer says the lamp was not the right fit for their TV and caused color problems on screen. [2'
- `23628` · partial · 'One reviewer says a Microsoft Surface 2 fits the sleeve very well, but the excerpts do not address the Surface Pro 3 with its keyboard attached. [1]\nReviewers report sleeve fit varies by laptop: one s'
- `39526` · partial · 'One reviewer says the coil pack was priced at half the price of a Dodge dealership, and after installation the check engine light was removed. [1]\nOne reviewer says the coil fits great and fixed a #3 '
- `65508` · partial · "Reviewers describe Jwoww's lotion as giving a noticeable tan after one or a few uses, and some say it outperforms other bronzers they tried. [1][2][3][4]\nOne reviewer said Jwoww made no visible differ"

## 6. 局限与下一步

- 金标每个 split 约 100 题，区间仍然较宽；结论只写表中的数值和区间。
- 金标的局限见第 1 节；结论支持率（逐条核对引用）不在本报告里。
- 全量上下文商品的题不进入检索指标，只出现在可答性评测里。
