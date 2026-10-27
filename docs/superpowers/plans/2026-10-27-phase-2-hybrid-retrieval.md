# 第 2 阶段：混合检索 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给定一件商品和一个问题，返回只属于这件商品的、带各通道排名和分数的证据候选；任一路检索出故障时另一路继续，并如实记录实际用了哪些通道。

**Architecture:** 新增 `reviewproof/retrieval/` 包。评论总量不超过阈值的商品直接返回全部块（全量上下文）；其余商品走两路召回：稠密通道（pgvector 在 SQL 里按商品精确排序）和 BM25 通道（`rank_bm25`，按商品建索引并缓存在内存），用 RRF 融合，取前 40 条交给 `bge-reranker-v2-m3` 重排，保留 8 条作证据。每个通道的成败、耗时和命中数写进结果；两路都失败时结果标记为服务故障，不能和"没有相关评论"混淆。最后提供一个命令行，用来查看某件商品某个问题的排序结果。

**Tech Stack:** Python 3.10+、psycopg 3 + pgvector、sentence-transformers（BGE-M3 向量、CrossEncoder 重排）、rank_bm25、unittest。

**Spec:** [docs/product_positioning_cn.md](../../product_positioning_cn.md)（F2 商品范围硬过滤、F3 混合检索 + 重排、6.3 检索指标、6.4 降级率）和 [docs/roadmap_cn.md](../../roadmap_cn.md)（第 2 节技术决定、第 3 节阶段 2）。

## Global Constraints

- 在分支 `phase-2-retrieval` 上工作，每个任务一次提交；本阶段结束后再决定合并和推送。
- 语法按 Python 3.10+；测试用 `unittest`，全量命令 `python -m unittest discover -s tests -b`；数据库测试继承 `tests.support.DatabaseTestCase`。
- 测试不下载、不加载任何真实模型：向量用 `tests.support.FakeEmbedder`，重排用本计划新增的 `tests.support.FakeReranker`。
- `requirements.txt` 本阶段只新增 `rank-bm25>=0.2.2,<1`。
- 环境变量前缀 `REVIEWPROOF_`；新增 `REVIEWPROOF_RERANK_MODEL`（默认 `BAAI/bge-reranker-v2-m3`）和 `REVIEWPROOF_FULL_CONTEXT_MAX_TOKENS`（默认 `6000`）。重排模型和向量模型共用 `REVIEWPROOF_EMBEDDING_DEVICE`，默认有 CUDA 用 CUDA，否则 CPU，不默认用 Apple MPS。
- 固定参数：RRF `k = 60`；每个通道取前 `50` 条；融合后前 `40` 条进重排；保留 `8` 条作证据。
- 全量上下文的判断依据是 `products.token_estimate`（第 1 阶段按每 4 个字符约 1 token 估算）：不超过阈值走全量上下文，否则走混合检索。当前 100 件商品里 28 件走全量上下文。
- 向量检索按商品精确排序，不建 ANN（HNSW、IVFFlat）索引；BM25 只在单件商品的块上计算，不引入 OpenSearch。
- 两路召回都必须在查询内按 `product_id` 过滤；结果中出现其他商品的块就是 bug。
- 检索语料只有 `chunks` 表。`questions` 表里的问题文本可以作为查询输入，但社区回答、`review_snippets` 和可答性标签不能进入候选，也不能影响排序。
- 证据单位是块（`chunk_id`），全量上下文和混合检索都返回块，后续阶段的引用统一用块 ID。
- 系统评论 ID 不是 Amazon 评论 ID，命令行输出里也不能这样称呼。
- 服务故障和"没有相关证据"必须能区分：`service_status` 为 `failed` 时，候选为空不代表评论里找不到答案。

## Review Focus

1. **查询里没有能匹配的英文词**（中文问题、只有标点、或者词都不在评论里）：BM25 通道应标记为 `skipped` 或命中 0 条，不算故障；稠密通道照常返回，`service_status` 仍是 `ok`。（任务 2、任务 5 测试）
2. **两路召回都失败**（读块失败、向量模型加载失败）：`service_status` 为 `failed`、候选为空，调用方能和"正常但没找到"区分开。（任务 5 测试）
3. **商品有块缺少当前模型的向量**（重新导入后没跑向量化，或换了向量模型）：稠密通道应标记失败并说明原因，不能悄悄只在部分块里检索。（任务 3、任务 5 测试）
4. **两件商品有一模一样的评论文本**：结果里只能有当前商品的块。（任务 2、任务 3、任务 5 测试）
5. **强制走混合检索的小商品，块数少于各个 K 值**（只有 3 个块）：返回不超过 3 条、没有重复、不报错。（任务 5 测试）

---

### Task 1: 配置与模型一致性检查

**Files:**
- Modify: `reviewproof/config.py`、`reviewproof/db.py`、`reviewproof/ingest/cli.py`、`requirements.txt`
- Test: `tests/test_config.py`、`tests/test_db_schema.py`、`tests/test_ingest_cli.py`

**Interfaces:**
- Consumes: 第 1 阶段的 `Settings`、`load_settings`、`verify_ingest`、`EMBEDDING_DIM`。
- Produces:
  - `Settings` 新增两个字段（放在已有字段之后，带默认值）：`rerank_model: str = "BAAI/bge-reranker-v2-m3"`、`full_context_max_tokens: int = 6000`。
  - `reviewproof.db.embedding_column_dim(conn) -> int`：返回数据库里 `chunks.embedding` 列实际的向量维度。
  - `verify_ingest(conn, manifest, require_embeddings: bool, embedding_model: str) -> dict`：`chunks_without_embedding` 改为统计"没有向量，或向量不是 `embedding_model` 算的"块。

这一任务处理第 1 阶段整体审查留下的两条：校验时没核对向量模型，数据库里已有表的向量维度没和代码核对。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_config.py
def test_retrieval_defaults_and_overrides(self):
    s = load_settings({})
    self.assertEqual((s.rerank_model, s.full_context_max_tokens), ("BAAI/bge-reranker-v2-m3", 6000))
    s = load_settings({"REVIEWPROOF_RERANK_MODEL": "m", "REVIEWPROOF_FULL_CONTEXT_MAX_TOKENS": "0"})
    self.assertEqual((s.rerank_model, s.full_context_max_tokens), ("m", 0))

def test_negative_threshold_rejected(self):
    with self.assertRaisesRegex(ValueError, "REVIEWPROOF_FULL_CONTEXT_MAX_TOKENS"):
        load_settings({"REVIEWPROOF_FULL_CONTEXT_MAX_TOKENS": "-1"})

# tests/test_db_schema.py（DatabaseTestCase）
def test_embedding_column_dim_matches_schema(self):
    self.assertEqual(embedding_column_dim(self.conn), EMBEDDING_DIM)

def test_ingest_refuses_database_with_other_dim(self):   # 放在 tests/test_ingest_cli.py 的 IngestCliDatabaseTests
    self.conn.execute("ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(8)"); self.conn.commit()
    self.addCleanup(self._restore_dim)   # ALTER ... TYPE vector(1024) 并 commit
    code, _, err = self.run_main(["--manifest", str(self.manifest_path)], {"REVIEWPROOF_DATABASE_URL": TEST_DATABASE_URL})
    self.assertEqual(code, 2)
    self.assertIn("vector(8)", err)

# tests/test_ingest_cli.py
def test_verify_counts_vectors_from_another_model_as_missing(self):
    self.run_main(["--manifest", str(self.manifest_path)], {"REVIEWPROOF_DATABASE_URL": TEST_DATABASE_URL})  # FakeEmbedder "fake"
    report = verify_ingest(self.conn, self.manifest, require_embeddings=True, embedding_model="other")
    self.assertEqual((report["chunks_without_embedding"], report["ok"]), (3, False))
```

已有的 `verify_ingest(...)` 调用补上 `embedding_model="fake"`。

- [ ] **Step 2:** 运行 `REVIEWPROOF_TEST_DATABASE_URL=postgresql://reviewproof:reviewproof@localhost:54329/reviewproof_test python -m unittest tests.test_config tests.test_db_schema tests.test_ingest_cli -b`，预期新测试 FAIL 或 ERROR（字段、函数、参数不存在）。
- [ ] **Step 3: 实现。**
  - `config.py`：`full_context_max_tokens` 用已有的 `_int` 读取，小于 0 时抛 `ValueError` 并写出变量名。
  - `db.py`：`embedding_column_dim` 查 `pg_attribute.atttypmod`（`attrelid = 'chunks'::regclass AND attname = 'embedding'`；对 pgvector 列，typmod 就是维度）。
  - `cli.py`：`apply_schema` 之后，若 `embedding_column_dim(conn) != EMBEDDING_DIM`，在 stderr 输出 `chunks.embedding is vector(<n>) but the code expects vector(1024)` 并返回 2；`verify_ingest` 调用时传 `settings.embedding_model`。
  - `requirements.txt` 新增 `rank-bm25>=0.2.2,<1`，并在本地 `.venv` 安装。
- [ ] **Step 4:** 运行同一命令，预期全部 PASS；再跑全量测试，预期全部 PASS。
- [ ] **Step 5: 提交** `feat: add retrieval settings and check vector model and dimension at ingest`

---

### Task 2: 检索数据类型与 BM25 通道

**Files:**
- Create: `reviewproof/retrieval/__init__.py`、`reviewproof/retrieval/types.py`、`reviewproof/retrieval/bm25.py`
- Test: `tests/test_bm25.py`

**Interfaces:**
- Consumes: `chunks` 表（`chunk_id`、`review_id`、`product_id`、`text`）。
- Produces（`types.py`）：

```python
@dataclass(frozen=True)
class ChunkRow:
    chunk_id: str
    review_id: str
    product_id: str
    text: str

@dataclass(frozen=True)
class Hit:            # 单个通道内的一条结果；名次就是在列表里的位置 + 1
    chunk: ChunkRow
    score: float
```

- Produces（`bm25.py`）：
  - `tokenize(text: str) -> list[str]`：`re.findall(r"[a-z0-9]+", text.lower())`，不去停用词、不做词干。
  - `load_product_chunks(conn, product_id: str) -> list[ChunkRow]`：按 `chunk_id` 升序。
  - `class BM25Cache(max_products: int = 64)`，方法 `search(conn, product_id: str, query: str, top_k: int) -> list[Hit]`。

规则：
- 只有和查询至少共享一个词的块才算命中；命中按 `BM25Okapi`（默认参数）分数降序，同分按 `chunk_id` 升序。不能用"分数大于 0"判断命中，因为块很少时 `rank_bm25` 会给常见词负的 IDF。
- 查询分词后为空时返回 `[]`，不查数据库；商品没有块（包括商品不存在）时也返回 `[]`，不建索引（`BM25Okapi` 遇到空语料会除零）。
- 缓存按商品存 `BM25Okapi` 和块列表，键是 `(product_id, 块数, 文本总长度)`。指纹用一条 `SELECT count(*), coalesce(sum(length(text)), 0) FROM chunks WHERE product_id = %s` 取得，指纹变了就重建。超过 `max_products` 时淘汰最久没用的。
- 读库后调用 `conn.rollback()` 结束只读事务（与第 1 阶段 `embed.py` 一致）。

- [ ] **Step 1: 写失败测试 `tests/test_bm25.py`**

```python
class TokenizeTests(unittest.TestCase):
    def test_lowercases_and_drops_punctuation(self):
        self.assertEqual(tokenize("Fits the 4th-gen iPad!"), ["fits", "the", "4th", "gen", "ipad"])

    def test_chinese_query_has_no_tokens(self):
        self.assertEqual(tokenize("声音大吗？"), [])

class BM25CacheTests(DatabaseTestCase):
    def setUp(self):
        super().setUp()
        load_dataset(self.conn, make_manifest("A", "B"),
                     {"A": make_product("A"), "B": make_product("B", ["Too loud for me.", "Cheap plastic."])}, {})
        self.cache = BM25Cache()

    def test_ranks_matching_chunk_first_within_product(self):
        hits = self.cache.search(self.conn, "A", "is it loud", top_k=50)
        self.assertEqual([h.chunk.chunk_id for h in hits], ["A-r00001-c00"])

    def test_identical_text_in_other_product_never_returned(self):
        hits = self.cache.search(self.conn, "A", "too loud", top_k=50)
        self.assertTrue(all(h.chunk.product_id == "A" for h in hits))

    def test_no_shared_terms_means_no_hits(self):
        self.assertEqual(self.cache.search(self.conn, "A", "waterproof", top_k=50), [])
        self.assertEqual(self.cache.search(self.conn, "A", "声音大吗？", top_k=50), [])

    def test_top_k_limits_hits(self):
        hits = self.cache.search(self.conn, "A", "battery loud ipad", top_k=2)
        self.assertEqual(len(hits), 2)

    def test_reimport_with_changed_text_rebuilds_index(self):
        self.cache.search(self.conn, "A", "loud", top_k=50)
        load_dataset(self.conn, make_manifest("A"), {"A": make_product("A", ["Quiet and waterproof."])}, {})
        self.assertEqual([h.chunk.chunk_id for h in self.cache.search(self.conn, "A", "waterproof", top_k=50)],
                         ["A-r00000-c00"])

    def test_unknown_product_returns_no_hits(self):
        self.assertEqual(self.cache.search(self.conn, "ZZZ", "loud", top_k=50), [])
```

- [ ] **Step 2:** 运行 `REVIEWPROOF_TEST_DATABASE_URL=... python -m unittest tests.test_bm25 -b`，预期 FAIL（模块不存在）。
- [ ] **Step 3: 实现 `types.py` 和 `bm25.py`。**
- [ ] **Step 4:** 运行同一命令，预期 PASS；全量测试 PASS。
- [ ] **Step 5: 提交** `feat: add per-product BM25 retrieval with an in-memory index cache`

---

### Task 3: 稠密通道

**Files:**
- Create: `reviewproof/retrieval/dense.py`
- Test: `tests/test_dense.py`

**Interfaces:**
- Consumes: `ChunkRow`、`Hit`（任务 2）；`Embedder`（第 1 阶段）。
- Produces:
  - `class IncompleteEmbeddingsError(RuntimeError)`：属性 `missing: int`。
  - `dense_search(conn, product_id: str, query_vector: Sequence[float], embedding_model: str, top_k: int) -> list[Hit]`。

规则：
- 先查这件商品有多少块"没有向量，或向量不是 `embedding_model` 算的"；大于 0 就抛 `IncompleteEmbeddingsError`，不做部分检索。
- 排序在 SQL 里完成：`WHERE product_id = %s ORDER BY embedding <=> %s, chunk_id LIMIT %s`，`score = 1 - 余弦距离`。查询向量用 `np.asarray(vec, dtype=np.float32)` 传入（和第 1 阶段写入一致）。不建任何向量索引。
- 读完后 `conn.rollback()`。

- [ ] **Step 1: 写失败测试 `tests/test_dense.py`**（继承 `DatabaseTestCase`）

```python
def setUp(self):
    super().setUp()
    load_dataset(self.conn, make_manifest("A", "B"),
                 {"A": make_product("A"), "B": make_product("B", ["Too loud for me.", "Cheap plastic."])}, {})
    embed_pending_chunks(self.conn, FakeEmbedder(), batch_size=10)
    self.vec = FakeEmbedder().embed_query("Too loud for me.")

def test_exact_text_ranks_first_with_cosine_one(self):
    hits = dense_search(self.conn, "A", self.vec, "fake", top_k=50)
    self.assertEqual(hits[0].chunk.chunk_id, "A-r00001-c00")
    self.assertAlmostEqual(hits[0].score, 1.0, places=5)
    self.assertEqual(len(hits), 3)

def test_identical_text_in_other_product_never_returned(self):
    hits = dense_search(self.conn, "A", self.vec, "fake", top_k=50)
    self.assertTrue(all(h.chunk.product_id == "A" for h in hits))

def test_top_k(self):
    self.assertEqual(len(dense_search(self.conn, "A", self.vec, "fake", top_k=2)), 2)

def test_missing_vector_raises_instead_of_partial_search(self):
    self.conn.execute("UPDATE chunks SET embedding = NULL WHERE chunk_id = 'A-r00002-c00'"); self.conn.commit()
    with self.assertRaises(IncompleteEmbeddingsError) as ctx:
        dense_search(self.conn, "A", self.vec, "fake", top_k=50)
    self.assertEqual(ctx.exception.missing, 1)

def test_vectors_from_another_model_raise(self):
    with self.assertRaises(IncompleteEmbeddingsError):
        dense_search(self.conn, "A", self.vec, "other-model", top_k=50)

def test_no_vector_index_exists(self):
    rows = self.conn.execute("SELECT indexdef FROM pg_indexes WHERE tablename = 'chunks'").fetchall()
    self.assertFalse(any("hnsw" in r[0] or "ivfflat" in r[0] for r in rows))
```

- [ ] **Step 2:** 运行 `REVIEWPROOF_TEST_DATABASE_URL=... python -m unittest tests.test_dense -b`，预期 FAIL。
- [ ] **Step 3: 实现 `dense.py`。**
- [ ] **Step 4:** 运行同一命令，预期 PASS；全量测试 PASS。
- [ ] **Step 5: 提交** `feat: add exact per-product dense retrieval over pgvector`

---

### Task 4: 重排模型

**Files:**
- Create: `reviewproof/retrieval/rerank.py`
- Modify: `reviewproof/embedding.py`（把 `_default_device` 改名为公开的 `default_device`，调用处同步）、`tests/support.py`（新增 `FakeReranker`）
- Test: `tests/test_rerank.py`

**Interfaces:**
- Produces:

```python
class Reranker(Protocol):
    model_name: str
    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...

class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 16, max_length: int = 512): ...
```

  - `tests.support.FakeReranker(fail: bool = False)`：`model_name = "fake-reranker"`；分数 = 查询和文本共享的 `tokenize` 词数（`float`）；`fail=True` 时 `score` 抛 `RuntimeError("FakeReranker failing")`。

规则：`CrossEncoderReranker` 第一次调用 `score` 时才 `from sentence_transformers import CrossEncoder` 并加载，参数 `CrossEncoder(model_name, device=device or default_device(), max_length=max_length)`；之后 `predict([(query, t) for t in texts], batch_size=..., show_progress_bar=False)`。`texts` 为空时直接返回 `[]`，不加载模型。返回条数和输入不一致时抛 `ValueError`。

- [ ] **Step 1: 写失败测试 `tests/test_rerank.py`**（离线，仿照 `tests/test_embedding.py`，用假的 `sentence_transformers` 模块）

```python
def test_lazy_load_device_and_pairs(self):
    # patch torch.cuda.is_available -> False
    reranker = CrossEncoderReranker("BAAI/bge-reranker-v2-m3")
    module.CrossEncoder.assert_not_called()
    self.assertEqual(reranker.score("q", ["a", "b"]), [0.5, 0.5])
    module.CrossEncoder.assert_called_once_with("BAAI/bge-reranker-v2-m3", device="cpu", max_length=512)
    self.assertEqual(model.predict.call_args.args[0], [("q", "a"), ("q", "b")])

def test_empty_texts_do_not_load_model(self):
    self.assertEqual(CrossEncoderReranker("m").score("q", []), [])
    module.CrossEncoder.assert_not_called()

def test_wrong_score_count_raises(self): ...   # predict 返回 1 个分数、输入 2 条 → ValueError

def test_fake_reranker_scores_shared_words(self):
    self.assertEqual(FakeReranker().score("is it loud", ["Too loud for me.", "Cheap plastic."]), [1.0, 0.0])
```

- [ ] **Step 2:** 运行 `python -m unittest tests.test_rerank -b`，预期 FAIL。
- [ ] **Step 3: 实现 `rerank.py`、`default_device` 改名、`FakeReranker`。**
- [ ] **Step 4:** 运行同一命令，预期 PASS；全量测试 PASS（`tests/test_embedding.py` 中引用 `_default_device` 的地方随改名更新）。
- [ ] **Step 5: 提交** `feat: add cross-encoder reranker`

---

### Task 5: 检索编排

**Files:**
- Create: `reviewproof/retrieval/retriever.py`
- Modify: `reviewproof/retrieval/types.py`、`reviewproof/retrieval/__init__.py`（导出 `retrieve`、`RetrievalParams`、`RetrievalResult`、`ProductNotFoundError`）、`tests/support.py`（`FakeEmbedder` 新增 `fail_queries: bool = False`，为真时 `embed_query` 抛 `RuntimeError`）
- Test: `tests/test_retriever.py`

**Interfaces:**
- Consumes: 任务 2 的 `BM25Cache`、`load_product_chunks`；任务 3 的 `dense_search`、`IncompleteEmbeddingsError`；任务 4 的 `Reranker`。
- Produces（加到 `types.py`）：

```python
@dataclass(frozen=True)
class RetrievalParams:
    rrf_k: int = 60
    channel_top_k: int = 50
    rerank_pool: int = 40
    evidence_k: int = 8
    full_context_max_tokens: int = 6000

@dataclass(frozen=True)
class Candidate:
    chunk: ChunkRow
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None

@dataclass(frozen=True)
class ChannelStatus:
    name: str                 # "dense" | "bm25" | "rerank"
    status: str               # "ok" | "failed" | "skipped"
    reason: str | None
    latency_ms: float
    hits: int

@dataclass(frozen=True)
class RetrievalResult:
    product_id: str
    query: str
    strategy: str             # "full_context" | "hybrid"
    mode: str                 # 实际成功的通道，如 "dense+bm25+rerank"、"bm25+rerank"、"dense+bm25"；全量上下文为 "full_context"；两路都失败为 "none"
    service_status: str       # "ok" | "degraded" | "failed"
    pool: tuple[Candidate, ...]        # 融合后、重排前的前 rerank_pool 条（第 5 阶段算 Recall@K 用）
    candidates: tuple[Candidate, ...]  # 最终证据，最多 evidence_k 条
    channels: tuple[ChannelStatus, ...]
```

- Produces（`retriever.py`）：
  - `class ProductNotFoundError(LookupError)`
  - `rrf_fuse(rankings: Mapping[str, Sequence[str]], k: int = 60) -> list[tuple[str, float]]`：`score(d) = Σ 1 / (k + rank)`，名次从 1 开始；按分数降序，同分按 `chunk_id` 升序。
  - `retrieve(conn, product_id: str, query: str, *, embedder: Embedder, embedding_model: str, reranker: Reranker | None, bm25_cache: BM25Cache, params: RetrievalParams = RetrievalParams(), strategy: str = "auto") -> RetrievalResult`

流程与规则：
1. `query.strip()` 为空、或 `strategy` 不在 `auto / hybrid / full_context` 中时抛 `ValueError`。读 `products` 行，不存在时抛 `ProductNotFoundError`（读商品行本身的数据库错误直接向上抛，由第 3 阶段的节点转成服务故障）。
2. `auto` 时 `token_estimate <= params.full_context_max_tokens` 走全量上下文：候选是这件商品的全部块（`load_product_chunks` 的顺序），`pool` 与 `candidates` 相同，三个通道都是 `skipped`、原因 `"full_context"`，`mode = "full_context"`，`service_status = "ok"`。
3. 混合检索：两个通道各自 `try`，任何异常都记成 `failed`，原因写 `f"{type(exc).__name__}: {exc}"`，并在继续之前调用 `conn.rollback()`，保证一个通道的 SQL 错误不会让另一个通道拿到已中止的事务。稠密通道 = `embedder.embed_query` + `dense_search(..., top_k=channel_top_k)`；BM25 通道在 `tokenize(query)` 为空时记 `skipped`、原因 `"no_query_terms"`，否则 `bm25_cache.search(..., top_k=channel_top_k)`。
4. 对成功通道的结果做 `rrf_fuse`，取前 `rerank_pool` 条组成 `pool`，每条都带上两个通道的名次和分数（没出现的那路为 `None`）。
5. 重排：`reranker` 为 `None` 时 `skipped`（原因 `"disabled"`）；`pool` 为空时 `skipped`（原因 `"no_candidates"`）；否则对 `pool` 打分，按重排分数降序（同分按 RRF 顺序）取前 `evidence_k` 条。重排出错时记 `failed`，按 RRF 顺序取前 `evidence_k` 条。
6. `service_status`：没有任何召回通道（稠密、BM25）是 `ok`，且其中至少一个是 `failed` → `failed`（候选为空，`mode = "none"`）。例如稠密失败、BM25 因中文查询被跳过，也是 `failed`。否则只要有任一通道（含重排）`failed` → `degraded`；其余情况为 `ok`。`skipped` 本身不算降级。
7. 最后检查所有候选的 `product_id` 都等于请求的商品，不等就抛 `AssertionError`（这是代码错误，不是可降级的情况）。

- [ ] **Step 1: 写失败测试 `tests/test_retriever.py`**

```python
class RRFTests(unittest.TestCase):
    def test_rrf_scores_and_order(self):
        fused = rrf_fuse({"dense": ["a", "b"], "bm25": ["b", "c"]}, k=60)
        self.assertEqual([d for d, _ in fused], ["b", "a", "c"])
        self.assertAlmostEqual(dict(fused)["b"], 1 / 62 + 1 / 61)

    def test_ties_break_by_id(self):
        self.assertEqual([d for d, _ in rrf_fuse({"dense": ["z"], "bm25": ["a"]})], ["a", "z"])

class RetrieveTests(DatabaseTestCase):
    def setUp(self):
        super().setUp()
        load_dataset(self.conn, make_manifest("A", "B"),
                     {"A": make_product("A"), "B": make_product("B", ["Too loud for me.", "Cheap plastic."])}, {})
        embed_pending_chunks(self.conn, FakeEmbedder(), batch_size=10)
        self.kw = dict(embedder=FakeEmbedder(), embedding_model="fake", reranker=FakeReranker(), bm25_cache=BM25Cache())

    def run_retrieve(self, query="Too loud for me.", strategy="hybrid", **overrides):
        return retrieve(self.conn, "A", query, strategy=strategy, **{**self.kw, **overrides})

    def test_small_product_uses_full_context_on_auto(self):
        r = self.run_retrieve(strategy="auto")
        self.assertEqual((r.strategy, r.mode, r.service_status), ("full_context", "full_context", "ok"))
        self.assertEqual([c.chunk.chunk_id for c in r.candidates], ["A-r00000-c00", "A-r00001-c00", "A-r00002-c00"])

    def test_threshold_zero_forces_hybrid_on_auto(self):
        r = self.run_retrieve(strategy="auto", params=RetrievalParams(full_context_max_tokens=0))
        self.assertEqual(r.strategy, "hybrid")

    def test_hybrid_all_channels_ok(self):
        r = self.run_retrieve()
        self.assertEqual((r.mode, r.service_status), ("dense+bm25+rerank", "ok"))
        self.assertEqual(r.candidates[0].chunk.chunk_id, "A-r00001-c00")
        top = r.candidates[0]
        self.assertEqual((top.dense_rank, top.bm25_rank), (1, 1))
        self.assertIsNotNone(top.rerank_score)

    def test_fewer_chunks_than_k_returns_each_chunk_once(self):
        r = self.run_retrieve(query="loud battery ipad")
        ids = [c.chunk.chunk_id for c in r.candidates]
        self.assertLessEqual(len(ids), 3)
        self.assertEqual(len(ids), len(set(ids)))

    def test_only_requested_product(self):
        r = self.run_retrieve()
        self.assertTrue(all(c.chunk.product_id == "A" for c in r.pool + r.candidates))

    def test_dense_failure_degrades_to_bm25(self):
        r = self.run_retrieve(embedder=FakeEmbedder(fail_queries=True))
        self.assertEqual((r.mode, r.service_status), ("bm25+rerank", "degraded"))
        self.assertEqual(self.status(r, "dense").status, "failed")
        self.assertTrue(r.candidates)

    def test_dense_sql_error_does_not_poison_bm25(self):
        r = self.run_retrieve(embedding_model="other-model")   # IncompleteEmbeddingsError
        self.assertEqual(self.status(r, "dense").reason.split(":")[0], "IncompleteEmbeddingsError")
        self.assertEqual(self.status(r, "bm25").status, "ok")

    def test_chinese_query_skips_bm25_without_degrading(self):
        r = self.run_retrieve(query="声音大吗？")
        self.assertEqual((self.status(r, "bm25").status, self.status(r, "bm25").reason), ("skipped", "no_query_terms"))
        self.assertEqual((r.mode, r.service_status), ("dense+rerank", "ok"))

    def test_both_channels_failed_is_service_failure_not_empty_answer(self):
        r = self.run_retrieve(query="声音大吗？", embedder=FakeEmbedder(fail_queries=True))
        # BM25 skipped + dense failed: no channel produced results
        self.assertEqual((r.mode, r.service_status, r.candidates), ("none", "failed", ()))

    def test_reranker_failure_falls_back_to_rrf_order(self):
        r = self.run_retrieve(reranker=FakeReranker(fail=True))
        self.assertEqual((r.mode, r.service_status), ("dense+bm25", "degraded"))
        self.assertEqual([c.chunk.chunk_id for c in r.candidates], [c.chunk.chunk_id for c in r.pool][:8])

    def test_reranker_disabled_is_not_degraded(self):
        r = self.run_retrieve(reranker=None)
        self.assertEqual((r.mode, r.service_status), ("dense+bm25", "ok"))

    def test_unknown_product_and_bad_input(self):
        with self.assertRaises(ProductNotFoundError):
            retrieve(self.conn, "ZZZ", "loud", **self.kw)
        with self.assertRaises(ValueError):
            self.run_retrieve(query="   ")
        with self.assertRaises(ValueError):
            self.run_retrieve(strategy="dense")

    def status(self, result, name):
        return next(c for c in result.channels if c.name == name)
```


- [ ] **Step 2:** 运行 `REVIEWPROOF_TEST_DATABASE_URL=... python -m unittest tests.test_retriever -b`，预期 FAIL。
- [ ] **Step 3: 实现 `retriever.py`，补充 `types.py` 和 `FakeEmbedder.fail_queries`。** 每个通道用 `time.perf_counter()` 记 `latency_ms`。
- [ ] **Step 4:** 运行同一命令，预期 PASS；全量测试 PASS。
- [ ] **Step 5: 提交** `feat: add hybrid retrieval with RRF, reranking, full-context strategy and channel status`

---

### Task 6: 检索命令行与真实数据验证

**Files:**
- Create: `reviewproof/retrieval/cli.py`、`reviewproof/retrieval/__main__.py`
- Modify: `CLAUDE.md`、`README.md`、`.env.example`
- Test: `tests/test_retrieval_cli.py`

**Interfaces:**
- Consumes: `retrieve`、`RetrievalParams`、`BM25Cache`、`CrossEncoderReranker`、`SentenceTransformerEmbedder`、`load_settings`。
- Produces: `main(argv: Sequence[str] | None = None, embedder_factory=None, reranker_factory=None) -> int`，命令 `python -m reviewproof.retrieval`。

参数：`--product ASIN`；`--query TEXT` 或 `--qid QID`（二选一；用 `--qid` 时从 `questions` 表取 `question_text` 和 `product_id`，此时 `--product` 可省略，给了但和题目所属商品不一致就报错返回 2）；`--strategy auto|hybrid|full_context`（默认 `auto`）；`--no-rerank`；`--json`。

输出：
- 默认是给人看的表格：第一行 `strategy / mode / service_status`；每个通道一行（状态、原因、耗时、命中数）；然后每条候选一行：名次、`chunk_id`、稠密名次、BM25 名次、RRF 分数、重排分数、文本前 100 个字符。
- `--json` 时把 `RetrievalResult` 用 `dataclasses.asdict` 输出成 JSON。
- 只输出检索结果，不输出社区回答、片段和可答性标签。
- 返回码：`service_status` 为 `failed` 时返回 1，否则 0；商品或题目不存在返回 2。
- `.env` 只在 `__main__.py` 里加载（与 `reviewproof/ingest/__main__.py` 相同）。

- [ ] **Step 1: 写失败测试 `tests/test_retrieval_cli.py`**（数据库测试；测试内的 `run_main(argv, embedder=FakeEmbedder())` 把 `REVIEWPROOF_DATABASE_URL` 设为测试库，通过两个 factory 参数注入 `embedder` 和 `FakeReranker()`，捕获 stdout/stderr，写法参照 `tests/test_ingest_cli.py`）

```python
def test_query_prints_table(self):
    code, out, _ = self.run_main(["--product", "A", "--query", "Too loud for me.", "--strategy", "hybrid"])
    self.assertEqual(code, 0)
    self.assertIn("dense+bm25+rerank", out)
    self.assertIn("A-r00001-c00", out)

def test_qid_reads_question_and_never_prints_answers(self):
    # setUp 写入一道 qid "q1" 的题，answers 含 "SECRET-ANSWER"，review_snippets 含 "SECRET-SNIPPET"
    code, out, _ = self.run_main(["--qid", "q1", "--json"])
    self.assertEqual(code, 0)
    self.assertEqual(json.loads(out)["product_id"], "A")
    self.assertNotIn("SECRET", out)

def test_qid_product_mismatch_exits_2(self):
    self.assertEqual(self.run_main(["--qid", "q1", "--product", "B"])[0], 2)

def test_unknown_product_exits_2(self):
    self.assertEqual(self.run_main(["--product", "ZZZ", "--query", "loud"])[0], 2)

def test_service_failure_exits_1(self):
    code, out, _ = self.run_main(["--product", "A", "--query", "声音大吗？", "--strategy", "hybrid"],
                                 embedder=FakeEmbedder(fail_queries=True))
    self.assertEqual(code, 1)
    self.assertIn("failed", out)
```

- [ ] **Step 2:** 运行 `REVIEWPROOF_TEST_DATABASE_URL=... python -m unittest tests.test_retrieval_cli -b`，预期 FAIL。
- [ ] **Step 3: 实现 `cli.py` 和 `__main__.py`。**
- [ ] **Step 4:** 运行同一命令，预期 PASS；全量测试 PASS。
- [ ] **Step 5: 真实数据验证。** 首次运行会从 Hugging Face 下载 `BAAI/bge-reranker-v2-m3`（约 2.3 GB），**下载前先征得用户同意**。用 dev 集的题目跑三次，并把输出摘要记进台账：
  - 一件 `small` 商品的题（`--strategy auto`）：预期 `mode = full_context`。
  - 一件 `large` 商品的题：预期 `mode = dense+bm25+rerank`、`service_status = ok`、8 条候选全部属于该商品；记录各通道耗时。
  - 同一件 `large` 商品，用中文改写的问题：预期 BM25 `skipped`、`mode = dense+rerank`。
  题目用 `SELECT q.qid, p.band FROM questions q JOIN products p USING (product_id) WHERE p.split = 'dev'` 挑选。不要用这几次结果下检索质量的结论，那是第 5 阶段的事。
- [ ] **Step 6: 更新文档。** `CLAUDE.md`：Commands 加检索命令，Architecture 加"phase 2: retrieval"一段（策略、通道、RRF 和重排参数、`service_status` 含义）。`README.md`：当前进度改为第 2 阶段完成，加一条检索命令示例。`.env.example`：加 `REVIEWPROOF_RERANK_MODEL` 和 `REVIEWPROOF_FULL_CONTEXT_MAX_TOKENS` 两行注释掉的默认值。
- [ ] **Step 7: 提交** `feat: add retrieval CLI and document phase 2`
