# 评测报告：pilot-test

## 1. 概要

- name: `pilot-test`
- split: `test`
- started_at: `2026-10-27T13:41:30+00:00`
- git_commit: `74094f18603505e6b09d76900d850ac90e90febd`
- dataset_id: `amazonqa-val-v1`
- source_sha256: `866095e6cce9abe5bc142e0ff1326737d57b5f24b19670123c6006d618bd941f`
- embedding_model: `BAAI/bge-m3`
- rerank_model: `BAAI/bge-reranker-v2-m3`
- llm_model: `claude-haiku-5-5`
- retrieval_params: `{'rrf_k': 60, 'channel_top_k': 50, 'rerank_pool': 40, 'evidence_k': 8, 'full_context_max_tokens': 6000}`
- max_rewrites: `1`
- seed: `20261027`
- questions_in_split: `528`
- sample: `50`
- systems: `['workflow', 'b0_no_evidence', 'b1_full_context']`
- cost_note: `Claude Haiku 5.5 prices as of 2026-10-27; each call priced at its own tier (over 100k input tokens is long-context)`

- **可答性银标**是 AmazonQA 分类器基于每题 10 个 BM25 片段做的判断，不是人工金标。换成完整评论后，部分"不可答"的题其实可答，所以对照银标的"误答"会被高估。
- **检索参考证据**是把 AmazonQA 为每题挑出的 10 个 BM25 片段对齐到评论块得到的，不代表这些块真的相关。检索分数衡量的是与 AmazonQA 的 BM25 结果的一致程度，天然偏向 BM25，不是人工金标。
- 人工金标评测在 5b 阶段进行。

## 2. 检索

共 528 题，计入指标 343 题；排除：{'full_context': 126, 'silver_unanswerable': 59}；各方案出错次数：{'bm25': 0, 'dense': 0, 'hybrid': 0, 'hybrid_rerank': 0}。每格为 均值 [95% 置信区间]。

| 方案 | n | recall@5 | recall@10 | recall@20 | recall@40 | hit@5 | mrr@10 | ndcg@5 |
|---|---|---|---|---|---|---|---|---|
| bm25 | 343 | 0.300 [0.286, 0.313] | 0.488 [0.469, 0.505] | 0.676 [0.656, 0.695] | 0.821 [0.800, 0.839] | 0.980 [0.965, 0.994] | 0.870 [0.839, 0.896] | 0.669 [0.640, 0.694] |
| dense | 343 | 0.165 [0.152, 0.177] | 0.259 [0.240, 0.276] | 0.401 [0.378, 0.424] | 0.574 [0.547, 0.600] | 0.825 [0.787, 0.860] | 0.631 [0.589, 0.669] | 0.379 [0.348, 0.406] |
| hybrid | 343 | 0.271 [0.258, 0.284] | 0.430 [0.409, 0.448] | 0.597 [0.575, 0.617] | 0.796 [0.776, 0.814] | 0.971 [0.953, 0.985] | 0.841 [0.808, 0.872] | 0.613 [0.583, 0.640] |
| hybrid_rerank | 343 | 0.230 [0.217, 0.244] | 0.377 [0.358, 0.397] | 0.571 [0.548, 0.593] | 0.796 [0.776, 0.814] | 0.913 [0.883, 0.942] | 0.745 [0.712, 0.779] | 0.517 [0.488, 0.546] |

每题平均有 10.7 个参考证据块，所以 Recall@5 的上限约为 0.47，不要把接近这个值的分数读成检索偏弱。

hybrid_rerank 减去其他方案（配对差值）：

| 对比 | hit@5 | ndcg@5 |
|---|---|---|
| hybrid_rerank − bm25 | -0.067 [-0.102, -0.035] | -0.151 [-0.184, -0.119] |
| hybrid_rerank − dense | 0.087 [0.050, 0.131] | 0.138 [0.113, 0.167] |
| hybrid_rerank − hybrid | -0.058 [-0.087, -0.029] | -0.095 [-0.121, -0.069] |

按品类的 hit@5：

| 品类 | bm25 | dense | hybrid | hybrid_rerank |
|---|---|---|---|---|
| Automotive | 0.986 [0.959, 1.000] | 0.838 [0.757, 0.919] | 0.986 [0.959, 1.000] | 0.919 [0.851, 0.973] |
| Beauty | 0.964 [0.929, 1.000] | 0.833 [0.750, 0.905] | 0.964 [0.917, 1.000] | 0.881 [0.810, 0.940] |
| Electronics | 0.968 [0.926, 1.000] | 0.800 [0.716, 0.874] | 0.958 [0.916, 0.989] | 0.905 [0.842, 0.958] |
| Home_and_Kitchen | 1.000 [1.000, 1.000] | 0.833 [0.756, 0.900] | 0.978 [0.944, 1.000] | 0.944 [0.889, 0.989] |

## 3. 可答性

"答了" = answered 或 partial；"拒答" = unanswerable 或不在服务范围。服务故障不进入分母，单独计数。

B1 只在混合检索商品的题上运行（全量上下文商品上它与工作流相同），所以和 B1 比较时请看"完整工作流（与 B1 同题）"一行，它只统计 B1 也跑了的题。

| 系统 | 计入/总数 | 服务故障 | 正确拒答率 | 误拒率 | 答复 F1 | 准确率 | 三态分布 | 服务故障被报成无法确认 |
|---|---|---|---|---|---|---|---|---|
| 完整工作流 | 50/50 | 0 | 0.636 [0.364, 0.909] | 0.128 [0.026, 0.231] | 0.883 [0.800, 0.943] | 0.820 [0.720, 0.920] | {'unanswerable': 12, 'answered': 25, 'partial': 13} | 0 |
| B0 不给证据 | 50/50 | 0 | 1.000 [1.000, 1.000] | 0.949 [0.872, 1.000] | 0.098 [0.000, 0.227] | 0.260 [0.160, 0.380] | {'unanswerable': 48, 'answered': 2} | 0 |
| B1 全量上下文 | 44/44 | 0 | 0.857 [0.571, 1.000] | 0.162 [0.054, 0.297] | 0.899 [0.814, 0.960] | 0.841 [0.727, 0.932] | {'unanswerable': 12, 'answered': 20, 'partial': 12} | 0 |
| 完整工作流（与 B1 同题） | 44/44 | 0 | 0.714 [0.286, 1.000] | 0.081 [0.000, 0.162] | 0.932 [0.861, 0.986] | 0.886 [0.795, 0.977] | {'unanswerable': 8, 'answered': 24, 'partial': 12} | 0 |

## 4. 成本与延迟

B1 只在混合检索商品的题上运行（全量上下文商品上它与工作流相同），所以和 B1 比较时请看"完整工作流（与 B1 同题）"一行，它只统计 B1 也跑了的题。

| 系统 | n | 总成本 (USD) | 每题成本 | P50 延迟 (s) | P95 延迟 (s) | 平均 LLM 调用 | 降级率 | 越界引用 |
|---|---|---|---|---|---|---|---|---|
| 完整工作流 | 50 | 0.0470 | 0.00094 | 15.2 | 34.4 | 2.9 | 0.000 | 0 |
| B0 不给证据 | 50 | 0.0042 | 0.00008 | 1.4 | 3.1 | 1.0 | 0.000 | 0 |
| B1 全量上下文 | 44 | 0.3360 | 0.00764 | 7.2 | 11.4 | 2.5 | 0.000 | 0 |
| 完整工作流（与 B1 同题） | 44 | 0.0427 | 0.00097 | 16.6 | 34.4 | 3.0 | 0.000 | 0 |

成本按 Claude Haiku 5.5 价格（2026-10-27）计算，每次调用按它自己的输入量选价格档（超过 10 万 token 为长上下文价格）。

## 5. 失败案例

以下为完整工作流的案例（每类最多 5 题）。对照银标的"误答"需人工核对：银标基于片段，完整评论可能确实能回答。

**误拒（银标可答，系统拒答）**

- `1172` · unanswerable · "This question is outside what reviews can answer: This question asks for a comparison with other Nespresso models, which the product's reviews cannot reliably answer."
- `39421` · unanswerable · "This question is outside what reviews can answer: Official product dimensions must come from the seller's product specifications, which I can't verify from customer reviews."
- `42050` · unanswerable · "The reviews don't confirm this."
- `81960` · unanswerable · 'This question is outside what reviews can answer: Reviews cannot tell where to buy a replacement part; that information is outside the scope of customer reviews.'
- `88580` · unanswerable · "Verdict: Can't confirm\nThe reviews don't confirm this."

**对照银标的误答（银标不可答，系统答了）——需人工核对**

- `12506` · answered · "The cooling pad has no controllable fan speed; one reviewer lists 'No Controllable Fan SpeedThe' as a con. [1]\nA reviewer says the product has no buttons, no lights, and no bells and whistles, which m"
- `31069` · partial · 'One reviewer measured all of the arms of the antenna at 31 inches each. [1]\nOne reviewer describes the antenna as having roughly 2-foot spokes extending in four directions. [2]\nOne reviewer reports th'
- `50882` · partial · 'Verdict: Yes\nOne reviewer says the duvet cover is very easy to clean and can simply be thrown in the wash. [1]\nOne reviewer says the cover has been much easier to wash, and notes that down comforters '
- `77335` · answered · "Verdict: Can't confirm\nMultiple reviewers confirm the speaker is an exact, direct-fit replacement for the front door of a 2002 Silverado. [1][2][3]\nA reviewer with a 2001 Silverado reports the speaker"

## 6. 局限与下一步

- 本报告是小规模试跑，样本量小，区间较宽；不据此下"提升了多少"的结论。
- 可答性和检索参考证据都来自 AmazonQA 银标，不是人工金标；5b 阶段做约 200 题人工金标后再给出结论。
- 全量上下文商品的题不进入检索指标，只出现在可答性评测里。
