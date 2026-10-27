# 评测报告：pilot-dev

## 1. 概要

- name: `pilot-dev`
- split: `dev`
- started_at: `2026-10-27T11:31:54+00:00`
- git_commit: `6988e6be52a6d50830dde27b58a7b79b54c28052`
- dataset_id: `amazonqa-val-v1`
- source_sha256: `866095e6cce9abe5bc142e0ff1326737d57b5f24b19670123c6006d618bd941f`
- embedding_model: `BAAI/bge-m3`
- rerank_model: `BAAI/bge-reranker-v2-m3`
- llm_model: `claude-haiku-5-5`
- retrieval_params: `{'rrf_k': 60, 'channel_top_k': 50, 'rerank_pool': 40, 'evidence_k': 8, 'full_context_max_tokens': 6000}`
- max_rewrites: `1`
- seed: `20261027`
- questions_in_split: `239`
- sample: `50`
- systems: `['workflow', 'b0_no_evidence', 'b1_full_context']`
- cost_note: `Claude Haiku 5.5 prices as of 2026-10-27; each call priced at its own tier (over 100k input tokens is long-context)`
- report_rebuilt_from_saved_rows: `table layout fix after the run; no rows recomputed`

- **可答性银标**是 AmazonQA 分类器基于每题 10 个 BM25 片段做的判断，不是人工金标。换成完整评论后，部分"不可答"的题其实可答，所以对照银标的"误答"会被高估。
- **检索参考证据**是把 AmazonQA 为每题挑出的 10 个 BM25 片段对齐到评论块得到的，不代表这些块真的相关。检索分数衡量的是与 AmazonQA 的 BM25 结果的一致程度，天然偏向 BM25，不是人工金标。
- 人工金标评测在 5b 阶段进行。

## 2. 检索

共 239 题，计入指标 145 题；排除：{'silver_unanswerable': 31, 'full_context': 63}；各方案出错次数：{'bm25': 0, 'dense': 0, 'hybrid': 0, 'hybrid_rerank': 0}。每格为 均值 [95% 置信区间]。

| 方案 | n | recall@5 | recall@10 | recall@20 | recall@40 | hit@5 | mrr@10 | ndcg@5 |
|---|---|---|---|---|---|---|---|---|
| bm25 | 145 | 0.299 [0.277, 0.319] | 0.466 [0.436, 0.498] | 0.644 [0.606, 0.681] | 0.815 [0.782, 0.846] | 0.986 [0.966, 1.000] | 0.872 [0.831, 0.912] | 0.665 [0.622, 0.704] |
| dense | 145 | 0.165 [0.144, 0.186] | 0.264 [0.233, 0.294] | 0.395 [0.355, 0.430] | 0.568 [0.523, 0.608] | 0.814 [0.745, 0.876] | 0.633 [0.565, 0.699] | 0.376 [0.330, 0.422] |
| hybrid | 145 | 0.256 [0.234, 0.277] | 0.416 [0.385, 0.447] | 0.586 [0.551, 0.620] | 0.773 [0.737, 0.805] | 0.938 [0.890, 0.972] | 0.842 [0.792, 0.889] | 0.584 [0.539, 0.628] |
| hybrid_rerank | 145 | 0.245 [0.221, 0.266] | 0.398 [0.367, 0.428] | 0.575 [0.539, 0.610] | 0.773 [0.737, 0.805] | 0.952 [0.910, 0.979] | 0.798 [0.740, 0.844] | 0.553 [0.504, 0.597] |

每题平均有 10.6 个参考证据块，所以 Recall@5 的上限约为 0.47，不要把接近这个值的分数读成检索偏弱。

hybrid_rerank 减去其他方案（配对差值）：

| 对比 | hit@5 | ndcg@5 |
|---|---|---|
| hybrid_rerank − bm25 | -0.034 [-0.076, 0.007] | -0.112 [-0.156, -0.072] |
| hybrid_rerank − dense | 0.138 [0.069, 0.207] | 0.177 [0.134, 0.220] |
| hybrid_rerank − hybrid | 0.014 [-0.028, 0.055] | -0.031 [-0.072, 0.007] |

按品类的 hit@5：

| 品类 | bm25 | dense | hybrid | hybrid_rerank |
|---|---|---|---|---|
| Automotive | 1.000 [1.000, 1.000] | 0.750 [0.571, 0.893] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| Beauty | 1.000 [1.000, 1.000] | 0.889 [0.778, 0.972] | 1.000 [1.000, 1.000] | 0.972 [0.917, 1.000] |
| Electronics | 0.951 [0.878, 1.000] | 0.756 [0.610, 0.878] | 0.878 [0.756, 0.976] | 0.902 [0.805, 0.976] |
| Home_and_Kitchen | 1.000 [1.000, 1.000] | 0.850 [0.725, 0.950] | 0.900 [0.800, 0.975] | 0.950 [0.875, 1.000] |

## 3. 可答性

"答了" = answered 或 partial；"拒答" = unanswerable 或不在服务范围。服务故障不进入分母，单独计数。

B1 只在混合检索商品的题上运行（全量上下文商品上它与工作流相同），所以和 B1 比较时请看"完整工作流（与 B1 同题）"一行，它只统计 B1 也跑了的题。

| 系统 | 计入/总数 | 服务故障 | 正确拒答率 | 误拒率 | 答复 F1 | 准确率 | 三态分布 | 服务故障被报成无法确认 |
|---|---|---|---|---|---|---|---|---|
| 完整工作流 | 50/50 | 0 | 0.250 [0.000, 0.500] | 0.237 [0.105, 0.368] | 0.763 [0.657, 0.861] | 0.640 [0.520, 0.780] | {'unanswerable': 12, 'answered': 23, 'partial': 15} | 0 |
| B0 不给证据 | 50/50 | 0 | 0.917 [0.750, 1.000] | 0.868 [0.737, 0.947] | 0.227 [0.050, 0.392] | 0.320 [0.200, 0.440] | {'unanswerable': 44, 'answered': 6} | 0 |
| B1 全量上下文 | 40/40 | 0 | 0.250 [0.000, 0.625] | 0.156 [0.031, 0.281] | 0.831 [0.712, 0.917] | 0.725 [0.575, 0.850] | {'answered': 21, 'partial': 12, 'unanswerable': 7} | 0 |
| 完整工作流（与 B1 同题） | 40/40 | 0 | 0.375 [0.125, 0.750] | 0.250 [0.125, 0.406] | 0.787 [0.667, 0.892] | 0.675 [0.525, 0.825] | {'unanswerable': 11, 'answered': 20, 'partial': 9} | 0 |

## 4. 成本与延迟

B1 只在混合检索商品的题上运行（全量上下文商品上它与工作流相同），所以和 B1 比较时请看"完整工作流（与 B1 同题）"一行，它只统计 B1 也跑了的题。

| 系统 | n | 总成本 (USD) | 每题成本 | P50 延迟 (s) | P95 延迟 (s) | 平均 LLM 调用 | 降级率 | 越界引用 |
|---|---|---|---|---|---|---|---|---|
| 完整工作流 | 50 | 0.0500 | 0.00100 | 15.7 | 28.8 | 3.0 | 0.000 | 0 |
| B0 不给证据 | 50 | 0.0049 | 0.00010 | 1.9 | 4.7 | 1.0 | 0.000 | 0 |
| B1 全量上下文 | 40 | 1.4798 | 0.03700 | 8.1 | 14.7 | 2.7 | 0.000 | 0 |
| 完整工作流（与 B1 同题） | 40 | 0.0368 | 0.00092 | 18.0 | 28.8 | 3.0 | 0.000 | 0 |

成本按 Claude Haiku 5.5 价格（2026-10-27）计算，每次调用按它自己的输入量选价格档（超过 10 万 token 为长上下文价格）。

## 5. 失败案例

以下为完整工作流的案例（每类最多 5 题）。对照银标的"误答"需人工核对：银标基于片段，完整评论可能确实能回答。

**误拒（银标可答，系统拒答）**

- `16553` · unanswerable · "The reviews don't confirm this."
- `47639` · unanswerable · 'This question is outside what reviews can answer: Comparisons with other products are outside what customer reviews of this item can answer.'
- `49083` · unanswerable · "The reviews don't confirm this."
- `51228` · unanswerable · 'This question is outside what reviews can answer: The question asks for a recommendation of a router type and compares it with other equipment (Motorola SB6121 and UniFi AP), which reviews of this sin'
- `54504` · unanswerable · 'This question is outside what reviews can answer: Warranty information must come from the seller, not customer reviews.'

**对照银标的误答（银标不可答，系统答了）——需人工核对**

- `37947` · answered · "Verdict: Can't confirm\nOne reviewer applied the adhesive vinyl to wood candles outdoors and said it adhered so well they looked painted on. [1]\nSeveral reviewers say the vinyl is very sticky and adher"
- `39526` · partial · "Verdict: Can't confirm\nUn comprador dice que la bobina costó la mitad que en un concesionario Dodge y que, tras instalarla, se apagó la luz de revisión del motor. [1]\nOtra reseña dice que la bobina en"
- `51047` · answered · 'The mirror tray is small: a reviewer said it was too small to center on a dresser and would not recommend it as a show piece. [1]\nOne reviewer said the tray is very small and would not look good on a '
- `57900` · partial · 'One reviewer lives in the mountains with only two poor local stations and did not have a clear view to the south, so they bought a 50-foot antenna cable for the house; the dock and play system worked '
- `60113` · partial · "Verdict: Can't confirm\nReviewers report the cover fits their Ford F-150 trucks well, including a 2012 F150 Lariat, and one reviewer's 2013 F150 fit as well. [1][2]\nOne reviewer said the cover fit abso"

## 6. 局限与下一步

- 本报告是小规模试跑，样本量小，区间较宽；不据此下"提升了多少"的结论。
- 可答性和检索参考证据都来自 AmazonQA 银标，不是人工金标；5b 阶段做约 200 题人工金标后再给出结论。
- 全量上下文商品的题不进入检索指标，只出现在可答性评测里。
