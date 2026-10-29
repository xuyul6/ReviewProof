# 评测报告：gold-dev-r2

## 1. 概要

- name: `gold-dev-r2`
- split: `dev`
- started_at: `2026-10-28T17:57:53+00:00`
- git_commit: `3a2b3371c9054ee816e7cccd2a97200c08d68e09`
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
- systems: `['workflow']`
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

| 系统 | 计入/总数 | 服务故障 | 正确拒答率 | 误拒率 | 答复 F1 | 准确率 | 三态分布 | 服务故障被报成无法确认 |
|---|---|---|---|---|---|---|---|---|
| 完整工作流 | 100/100 | 0 | 0.522 [0.304, 0.739] | 0.052 [0.013, 0.117] | 0.907 [0.854, 0.952] | 0.850 [0.770, 0.920] | {'answered': 49, 'unanswerable': 16, 'partial': 35} | 0 |
| B2 AmazonQA 自带分类器 | 100/100 | 0 | 0.565 [0.348, 0.783] | 0.156 [0.078, 0.234] | 0.855 [0.791, 0.910] | 0.780 [0.700, 0.860] | {'answered': 75, 'unanswerable': 25} | 0 |

## 3b. 三态混淆矩阵与是非题结论

行是金标，列是系统输出；服务故障不计入。三态一致率 = 系统三态与金标完全相同的比例。

**完整工作流**（n=100，三态一致率 0.670 [0.570, 0.760]）

| 金标 \ 系统 | answered | partial | unanswerable |
|---|---|---|---|
| answered | 42 | 11 | 1 |
| partial | 7 | 13 | 3 |
| unanswerable | 0 | 11 | 12 |

是非题结论准确率（金标结论为 yes / no / mixed 的是非题；系统没给结论算错）：

| 系统 | n | 准确率 |
|---|---|---|
| 完整工作流 | 12 | 0.500 [0.250, 0.750] |

## 3c. 一致率

| 对比 | n | Cohen's kappa | 一致比例 | 备注 |
|---|---|---|---|---|
| 金标 vs AmazonQA 银标（二值） | 100 | 0.397 | 0.780 | 银标基于 10 个 BM25 片段 |
| 子 agent 预标 vs 人工终审（三态） | 100 | 1.000 | 1.000 | 终审改动了 0.030 的题（任一项不同即算） |
| 人工盲标 vs 子 agent 预标（三态） | 20 | 0.593 | 0.750 | 人和子 agent 之间，不是两位人工之间 |

## 3d. 按评论量分段（混合检索商品，工作流 vs B1）

只统计 B1 也跑了的题，按商品评论总 token 数分档；数值是三态一致率。用来判断全量上下文阈值该不该调。

没有可比的题。

## 4. 成本与延迟

| 系统 | n | 总成本 (USD) | 每题成本 | P50 延迟 (s) | P95 延迟 (s) | 平均 LLM 调用 | 降级率 | 越界引用 |
|---|---|---|---|---|---|---|---|---|
| 完整工作流 | 100 | 0.1151 | 0.00115 | 11.3 | 20.1 | 3.1 | 0.000 | 0 |

成本按 Claude Haiku 5.5 价格（2026-10-27）计算，每次调用按它自己的输入量选价格档（超过 10 万 token 为长上下文价格）。

## 5. 失败案例

以下为完整工作流的案例（每类最多 5 题），对照人工金标。

**误拒（金标可答，系统拒答）**

- `10844` · unanswerable · "The reviews don't confirm this."
- `18315` · unanswerable · "This question is outside what reviews can answer: This question asks whether the seller carries a product (bags) in stock, which is a question about the seller's business rather than about using or ow"
- `68384` · unanswerable · "Verdict: Can't confirm\nThe reviews don't confirm this."
- `81986` · unanswerable · "Verdict: Can't confirm\nThe reviews don't confirm this."

**误答（金标不可答，系统答了）**

- `12206` · partial · 'Verdict: Yes\nSeveral reviewers say the lamp fit their JVC rear projection TV perfectly and worked like the original. [1][2][3]\nOther reviewers say the lamp did not fit their JVC rear projection TV, or'
- `23628` · partial · 'Reviewers report that the sleeve fits laptops of various sizes well, including 11.5 to 15.6 inch laptops, and some note a snug or perfect fit. [1][2][3][4]\nOne reviewer says a Surface 2 fits the sleev'
- `36942` · partial · 'Verdict: Yes\nOne reviewer says the iron constantly turns itself off while in use after it stopped working, which they attribute possibly to stuck buttons. [1]\nA reviewer says that after months of use '
- `39526` · partial · "Verdict: Can't confirm\nOne reviewer says the coil pack fits great. [1]\nA reviewer says that after installing the coil pack, the check engine light was removed. [2]\nA reviewer says the coil pack cost h"
- `40951` · partial · "Verdict: Yes\nReviewers say the receiver attaches to the camera's hot shoe and the remote's receiver can be mounted there, but a reviewer who wanted to use the camera's flash attachment could not use t"

## 6. 局限与下一步

- 金标每个 split 约 100 题，区间仍然较宽；结论只写表中的数值和区间。
- 金标的局限见第 1 节；结论支持率（逐条核对引用）不在本报告里。
- 全量上下文商品的题不进入检索指标，只出现在可答性评测里。
