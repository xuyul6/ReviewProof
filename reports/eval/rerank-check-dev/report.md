# reranker 分数检查：rerank-check-dev

reranker 给出的最高分，能不能区分评论能回答和不能回答的题？这次检查用来决定第 6 阶段要不要训练可答性模型。没有调用 LLM。

## 1. 做法

- 题目：金标 dev 集 100 题（answered 54、partial 23、unanswerable 23），answered 和 partial 都算可答。打分失败 0 题，不计入任何指标。
- 打分范围和系统一致：全量上下文商品给每个块打分；混合检索商品只给重排池里的 40 个块打分。
- 查询用题目原文，不是工作流规划后的英文查询；英文题两者基本相同，中文翻译和改写查询不在这次检查里。
- 工作流和 B2（AmazonQA 银标）的结果取自 `reports/eval/gold-dev/per_question.jsonl`，是同一批题此前的金标评测。
- 区间是 95% bootstrap，可答和不可答分开重采样，1000 次，种子 20261027。
- reranker：`BAAI/bge-reranker-v2-m3`；git_commit：`587fa680995a693d1b5f8303485a2b043e1fe15e-dirty`。

## 2. 最高分区分可答和不可答（AUC）

| 范围 | AUC | 95% 区间 | 可答 | 不可答 |
|---|---|---|---|---|
| 全部 | 0.802 | 0.689–0.892 | 77 | 23 |
| 混合检索 | 0.815 | 0.677–0.929 | 56 | 14 |
| 全量上下文 | 0.751 | 0.561–0.905 | 21 | 9 |

最高分的中位数：可答 0.336，不可答 0.015。

## 3. 和工作流比可答 / 不可答二分的准确率

| 判定方式 | 准确率 |
|---|---|
| 工作流 | 0.850 |
| 只看最高分：最高分 ≥ 0.0031 判可答（阈值在同一批题上挑的，偏乐观） | 0.830 |
| B2（AmazonQA 银标） | 0.780 |
| 全部判可答 | 0.770 |

## 4. 工作流拒答的题

工作流拒答 30 题：误拒 11（金标 answered 3、partial 8），正确拒答 19。另有误答 4 题（金标不可答，工作流作答）。

在拒答的题里，最高分区分误拒和正确拒答：

| 范围 | AUC | 95% 区间 | 误拒 | 正确拒答 |
|---|---|---|---|---|
| 工作流拒答的题 | 0.608 | 0.364–0.837 | 11 | 19 |

下表是工作流拒答和误答的题，按最高分从高到低。最后一列是金标 2 级证据块在打分列表里的最好名次；“—”表示金标没有 2 级块。

| qid | 最高分 | 金标 | 工作流 | 检索策略 | 2 级证据最好名次 |
|---|---|---|---|---|---|
| 1505 | 0.960 | partial | unanswerable | hybrid | 1 |
| 65508 | 0.932 | unanswerable | partial | hybrid | — |
| 27383 | 0.580 | answered | unanswerable | hybrid | 1 |
| 40951 | 0.476 | unanswerable | unanswerable | full_context | — |
| 38107 | 0.327 | partial | unanswerable | hybrid | 2 |
| 52521 | 0.267 | unanswerable | unanswerable | hybrid | — |
| 7244 | 0.160 | unanswerable | unanswerable | hybrid | — |
| 83461 | 0.155 | unanswerable | unanswerable | hybrid | — |
| 12206 | 0.145 | unanswerable | partial | hybrid | — |
| 87713 | 0.109 | answered | unanswerable | hybrid | 10 |
| 73170 | 0.086 | partial | 超出范围 | hybrid | 32 |
| 3291 | 0.083 | unanswerable | unanswerable | full_context | — |
| 77003 | 0.065 | unanswerable | unanswerable | full_context | — |
| 56496 | 0.051 | unanswerable | unanswerable | full_context | — |
| 37474 | 0.043 | partial | unanswerable | hybrid | 不在打分范围 |
| 39526 | 0.020 | unanswerable | partial | full_context | — |
| 23628 | 0.018 | unanswerable | partial | hybrid | — |
| 23674 | 0.017 | partial | unanswerable | full_context | 10 |
| 53648 | 0.015 | unanswerable | unanswerable | hybrid | — |
| 83254 | 0.009 | unanswerable | unanswerable | full_context | — |
| 23528 | 0.009 | partial | unanswerable | hybrid | 4 |
| 36942 | 0.003 | unanswerable | unanswerable | hybrid | — |
| 85706 | 0.002 | unanswerable | unanswerable | hybrid | — |
| 9762 | 0.002 | unanswerable | unanswerable | full_context | — |
| 42477 | 0.002 | unanswerable | unanswerable | hybrid | — |
| 2932 | 0.001 | unanswerable | unanswerable | full_context | — |
| 71487 | 0.001 | unanswerable | unanswerable | hybrid | — |
| 70224 | 0.001 | unanswerable | 超出范围 | hybrid | — |
| 68384 | 0.001 | partial | unanswerable | full_context | 4 |
| 81986 | 0.000 | partial | unanswerable | full_context | 8 |
| 66255 | 0.000 | unanswerable | unanswerable | hybrid | — |
| 71175 | 0.000 | unanswerable | 超出范围 | hybrid | — |
| 10844 | 0.000 | answered | unanswerable | hybrid | 16 |
| 19796 | 0.000 | unanswerable | 超出范围 | full_context | — |

这是 dev 集上的一次检查：不可答只有 23 题，区间很宽。
