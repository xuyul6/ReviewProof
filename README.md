# ReviewProof

购前单商品评论问答：选定一件商品，用这件商品的真实评论回答购买前的问题。每个回答标明"能回答 / 部分能回答 / 评论里找不到"，并给出引用的评论原文。数据来自 [AmazonQA](https://github.com/amazonqa/amazonqa)。

- 做什么、不做什么：[项目定位](docs/product_positioning_cn.md)
- 怎么做、分几个阶段：[改动方案](docs/roadmap_cn.md)

## 当前进度

- 第 1 阶段（数据底座）：从 AmazonQA 抽样 100 件商品（4 个品类），共 22,433 条评论、24,599 个评论块和 767 道评测题，存入 PostgreSQL + pgvector，每个块都有 BGE-M3 向量。
- 第 2 阶段（混合检索）：给定商品和问题，评论少的商品直接返回全部评论块，其余商品用向量检索和 BM25 两路召回、RRF 融合、BGE 重排，取 8 条证据。任一路出故障时另一路继续，并记录实际用了哪些通道。

问答工作流、接口和界面在后续阶段实现。

## 本地运行

需要 Python 3.10+ 和 Docker。

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # 填入 ANTHROPIC_API_KEY（第 3 阶段起用于生成回答）
docker compose up -d db           # 本地 Postgres 16 + pgvector，端口 54329
```

把 AmazonQA 验证集的 `val-qar_products.jsonl` 和 `val-qar.jsonl` 下载到 `data/`，然后导入：

```bash
python -m reviewproof.ingest --manifest samples/manifest_v1.json
```

命令会先核对数据文件的 SHA-256，再写入商品、评论、评论块和题目，最后计算向量并校验。可以重复运行：已经写入的数据不会重复，中途中断后再运行会接着计算剩下的向量。首次运行会从 Hugging Face 下载 `BAAI/bge-m3`（约 2.3 GB）。

查看某道题的检索结果（第一次运行会下载重排模型 `BAAI/bge-reranker-v2-m3`，约 2.3 GB；加 `--no-rerank` 可以跳过重排）：

```bash
python -m reviewproof.retrieval --qid 17020
python -m reviewproof.retrieval --product B00007E7C8 --query "Does it have volume control?"
```

## 测试

```bash
python -m unittest discover -s tests -b
REVIEWPROOF_TEST_DATABASE_URL=postgresql://reviewproof:reviewproof@localhost:54329/reviewproof_test python -m unittest discover -s tests -b
```

第一条只跑不需要数据库的测试；第二条连同数据库测试一起跑，只会清空名字以 `_test` 结尾的测试库。
