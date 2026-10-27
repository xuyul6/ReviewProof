# 第 1 阶段：数据底座 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把按固定清单抽样的 100 件 AmazonQA 商品（完整评论 + 评测用问题）导入本地 PostgreSQL，评论切块后用 BGE-M3 向量化，一条命令可重复执行。

**Architecture:** 新代码放在新包 `reviewproof/`，旧 `bridge/` 本阶段不动。离线流水线：抽样脚本生成 `samples/manifest_v1.json` → 读取器按清单读取并合并商品记录、生成系统评论 ID → 切分器生成带偏移量的块 → 写库（psycopg 3，幂等 upsert）→ 向量化（可断点续跑）→ 校验。问题、社区回答和可答性银标只进 `questions` 表。

**Tech Stack:** Python 3.10+、psycopg 3、pgvector（PostgreSQL 16 扩展和 Python 适配器）、sentence-transformers（BGE-M3）、rank_bm25（本阶段不用）、unittest、Docker Compose、GitHub Actions。

**Spec:** [docs/product_positioning_cn.md](../../product_positioning_cn.md)（第 5 节数据方案）和 [docs/roadmap_cn.md](../../roadmap_cn.md)（第 2 节技术决定、第 3 节阶段 1）。

## Global Constraints

- 在分支 `phase-1-data` 上工作，每个任务一次提交；本阶段结束后再决定合并和推送。
- 新代码只放在 `reviewproof/`、`tests/`、`samples/`、`docker/`、`.github/` 和根目录配置文件；本阶段不修改 `bridge/` 和 `app/`。
- 语法按 Python 3.10+（`list[str]`、`X | None`）；测试用 `unittest`，全量命令 `python -m unittest discover -s tests -v`，旧测试必须继续通过。
- `requirements.txt` 只新增 `psycopg[binary]>=3.2,<4` 和 `pgvector>=0.3,<1`。
- 环境变量前缀 `REVIEWPROOF_`。
- 纯逻辑测试必须离线；数据库测试只在设置 `REVIEWPROOF_TEST_DATABASE_URL` 时运行，否则跳过。
- 测试辅助代码放在 `tests/support.py`，用 `from tests.support import ...` 导入（单测命令 `python -m unittest tests.test_xxx` 也能运行）。
- `questionText`、`answers`、`is_answerable`、`review_snippets` 只写进 `questions` 表，不进入 `reviews`、`chunks`，也不参与向量化。
- 评论 ID 是系统生成的：`{asin}-r{index:05d}`；块 ID 是 `{review_id}-c{index:02d}`。代码、注释和文档都不能把它们称为 Amazon 评论 ID。
- 文本规范化统一用 `reviewproof.text.normalize_text`（`html.unescape` → 连续空白折叠成一个空格 → 去掉首尾空白）。后续阶段的片段对齐也必须用它。
- `data/` 不进 git；`samples/manifest_v1.json` 进 git。
- `val-qar_products.jsonl` 的 SHA-256：`866095e6cce9abe5bc142e0ff1326737d57b5f24b19670123c6006d618bd941f`。
- 向量维度固定 1024（BGE-M3），写在 schema 和配置里，两者不一致时导入命令报错。
- `dataset_id` 为 `amazonqa-val-v1`；抽样随机种子 `20261027`。

## Review Focus

1. **HTML 实体：** 10,117 行评论含 `&#…` 实体，片段里也有 `&quot;`。存储和切分要用解码后的文本，原文另存一列。（任务 3 测试）
2. **空评论：** 前 3,000 件商品里有 109 条空评论。要跳过，并且不占用评论序号。（任务 3 测试）
3. **同一 ASIN 出现在多条记录中：** 至少 13 个。要合并、去掉重复评论，ID 保持稳定。（任务 3 测试）
4. **向量化中途失败：** 重跑时只补没完成的块，已写入的不重复处理。（任务 7 测试）
5. **清单与本地文件不一致**（比如重新下载了别的版本）：导入前校验 SHA-256，不一致就报错退出，不写库。（任务 8 测试）

---

### Task 1: 包骨架、配置、本地数据库和 CI

**Files:**
- Create: `reviewproof/__init__.py`（空）、`reviewproof/config.py`
- Create: `docker-compose.yml`、`docker/initdb/01-test-db.sql`、`.github/workflows/tests.yml`
- Modify: `requirements.txt`（追加两行依赖）
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `Settings`（frozen dataclass）：`database_url: str`、`embedding_model: str`、`embedding_dim: int`、`embedding_batch_size: int`
  - `load_settings(env: Mapping[str, str] | None = None) -> Settings`。`env=None` 时读 `os.environ`；不在这里加载 `.env`。

- [ ] **Step 1: 写失败测试 `tests/test_config.py`**

```python
class SettingsTests(unittest.TestCase):
    def test_defaults(self):
        s = load_settings({})
        self.assertEqual(s.database_url, "postgresql://reviewproof:reviewproof@localhost:54329/reviewproof")
        self.assertEqual((s.embedding_model, s.embedding_dim, s.embedding_batch_size), ("BAAI/bge-m3", 1024, 32))

    def test_env_overrides(self):
        s = load_settings({"REVIEWPROOF_DATABASE_URL": "postgresql://x/y", "REVIEWPROOF_EMBEDDING_BATCH_SIZE": "8"})
        self.assertEqual((s.database_url, s.embedding_batch_size), ("postgresql://x/y", 8))

    def test_bad_int_names_the_variable(self):
        with self.assertRaisesRegex(ValueError, "REVIEWPROOF_EMBEDDING_DIM"):
            load_settings({"REVIEWPROOF_EMBEDDING_DIM": "abc"})
```

- [ ] **Step 2:** 运行 `python -m unittest tests.test_config -v`，预期 FAIL（`ModuleNotFoundError: reviewproof`）。
- [ ] **Step 3: 实现 `reviewproof/config.py`。** 环境变量：`REVIEWPROOF_DATABASE_URL`、`REVIEWPROOF_EMBEDDING_MODEL`、`REVIEWPROOF_EMBEDDING_DIM`、`REVIEWPROOF_EMBEDDING_BATCH_SIZE`，默认值见测试。
- [ ] **Step 4:** 运行同一命令，预期 PASS。
- [ ] **Step 5: 本地数据库与 CI。**
  - `docker-compose.yml`：服务 `db`，镜像 `pgvector/pgvector:pg16`；`POSTGRES_USER`、`POSTGRES_PASSWORD`、`POSTGRES_DB` 都是 `reviewproof`；端口 `54329:5432`；命名卷 `reviewproof-pgdata`；`./docker/initdb` 挂到 `/docker-entrypoint-initdb.d`；健康检查 `pg_isready -U reviewproof`。
  - `docker/initdb/01-test-db.sql`：`CREATE DATABASE reviewproof_test;`。本地测试库 URL 为 `postgresql://reviewproof:reviewproof@localhost:54329/reviewproof_test`。
  - `.github/workflows/tests.yml`：push 和 pull_request 触发；`services.postgres` 用 `pgvector/pgvector:pg16`（`POSTGRES_USER=reviewproof`、`POSTGRES_PASSWORD=reviewproof`、`POSTGRES_DB=reviewproof_test`，端口 5432，健康检查 `pg_isready -U reviewproof`）；Python 3.12；`pip install -r requirements.txt`；以 `REVIEWPROOF_TEST_DATABASE_URL=postgresql://reviewproof:reviewproof@localhost:5432/reviewproof_test` 运行全量测试。
  - `requirements.txt` 追加 `psycopg[binary]>=3.2,<4` 和 `pgvector>=0.3,<1`，然后 `pip install -r requirements.txt`。
- [ ] **Step 6:** 运行 `docker compose up -d db`，再运行 `docker compose exec db psql -U reviewproof -l`，预期列表里有 `reviewproof` 和 `reviewproof_test`。然后跑全量测试，预期全部 PASS。
- [ ] **Step 7: 提交** `feat: add reviewproof package skeleton, settings, local Postgres and CI`

### Task 2: 数据库 schema 与连接

**Files:**
- Create: `reviewproof/db.py`、`reviewproof/schema.sql`
- Create: `tests/support.py`（`DatabaseTestCase`）
- Test: `tests/test_db_schema.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `EMBEDDING_DIM = 1024`
  - `ALL_TABLES = ("questions", "chunks", "reviews", "products", "datasets")`
  - `connect(database_url: str) -> psycopg.Connection`：执行 `CREATE EXTENSION IF NOT EXISTS vector`、提交，然后 `pgvector.psycopg.register_vector(conn)`。
  - `apply_schema(conn) -> None`：执行 `schema.sql` 并提交，可以重复执行。
  - `tests.support.DatabaseTestCase`：带 `@unittest.skipUnless(TEST_DATABASE_URL, "set REVIEWPROOF_TEST_DATABASE_URL to run database tests")`。`setUp` 中连接、`apply_schema`、`TRUNCATE` 全部表（`CASCADE`）后提交；`tearDown` 关闭连接；连接放在 `self.conn`。

`reviewproof/schema.sql` 的内容如下（这是规格，照抄）：

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS datasets (
  dataset_id TEXT PRIMARY KEY, source_file TEXT NOT NULL, source_sha256 TEXT NOT NULL,
  manifest JSONB NOT NULL, loaded_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS products (
  product_id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL REFERENCES datasets,
  category TEXT NOT NULL, title TEXT,
  split TEXT NOT NULL CHECK (split IN ('dev', 'test')),
  band TEXT NOT NULL CHECK (band IN ('large', 'medium', 'small')),
  review_count INT NOT NULL, chunk_count INT NOT NULL, token_estimate INT NOT NULL);
CREATE TABLE IF NOT EXISTS reviews (
  review_id TEXT PRIMARY KEY, product_id TEXT NOT NULL REFERENCES products ON DELETE CASCADE,
  review_index INT NOT NULL, raw_text TEXT NOT NULL, text TEXT NOT NULL,
  helpful_yes INT NOT NULL, helpful_total INT NOT NULL,
  source_line INT NOT NULL, source_position INT NOT NULL,
  UNIQUE (product_id, review_index));
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY, review_id TEXT NOT NULL REFERENCES reviews ON DELETE CASCADE,
  product_id TEXT NOT NULL REFERENCES products ON DELETE CASCADE,
  chunk_index INT NOT NULL, text TEXT NOT NULL,
  start_offset INT NOT NULL, end_offset INT NOT NULL, token_estimate INT NOT NULL,
  embedding vector(1024), embedding_model TEXT);
CREATE INDEX IF NOT EXISTS chunks_product_idx ON chunks (product_id);
CREATE TABLE IF NOT EXISTS questions (
  qid TEXT PRIMARY KEY, product_id TEXT NOT NULL REFERENCES products ON DELETE CASCADE,
  question_text TEXT NOT NULL, question_type TEXT NOT NULL,
  is_answerable_silver BOOLEAN NOT NULL, answers JSONB NOT NULL, review_snippets JSONB NOT NULL);
CREATE INDEX IF NOT EXISTS questions_product_idx ON questions (product_id);
```

- [ ] **Step 1: 写失败测试 `tests/test_db_schema.py`**（继承 `DatabaseTestCase`）

```python
def test_apply_schema_is_idempotent_and_creates_tables(self):
    apply_schema(self.conn)  # setUp 已执行过一次
    names = {r[0] for r in self.conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")}
    self.assertTrue(set(ALL_TABLES) <= names)

def test_vector_extension_and_dimension(self):
    self.assertEqual(self.conn.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'").fetchone()[0], "vector")
    typ = self.conn.execute("SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                            "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'").fetchone()[0]
    self.assertEqual(typ, f"vector({EMBEDDING_DIM})")
```

- [ ] **Step 2:** 运行 `REVIEWPROOF_TEST_DATABASE_URL=postgresql://reviewproof:reviewproof@localhost:54329/reviewproof_test python -m unittest tests.test_db_schema -v`，预期 FAIL（`reviewproof.db` 不存在）。不设置这个变量时运行，预期 2 个测试都 skipped。
- [ ] **Step 3: 实现 `reviewproof/db.py` 和 `tests/support.py`**，`schema.sql` 用 `importlib.resources` 读取。
- [ ] **Step 4:** 带环境变量运行，预期 PASS。
- [ ] **Step 5: 提交** `feat: add Postgres schema and connection helpers`

### Task 3: AmazonQA 读取、合并与系统评论 ID

**Files:**
- Create: `reviewproof/text.py`、`reviewproof/ingest/__init__.py`（空）、`reviewproof/ingest/amazonqa.py`
- Modify: `tests/support.py`（新增 `write_jsonl(path: Path, rows: Iterable[dict]) -> Path`）
- Test: `tests/test_text.py`、`tests/test_amazonqa.py`

**Interfaces:**
- Produces:
  - `normalize_text(text: str) -> str`；`dedupe_key(text: str) -> str`（即 `normalize_text(text).lower()`）
  - `Review`（frozen dataclass）：`review_id: str, product_id: str, review_index: int, raw_text: str, text: str, helpful_yes: int, helpful_total: int, source_line: int, source_position: int`
  - `Product`（frozen dataclass）：`product_id: str, category: str, reviews: tuple[Review, ...]`
  - `Question`（frozen dataclass）：`qid: str, product_id: str, question_text: str, question_type: str, is_answerable: bool, answers: tuple[dict, ...], review_snippets: tuple[str, ...]`
  - `review_id_for(product_id: str, review_index: int) -> str` → `f"{product_id}-r{review_index:05d}"`
  - `load_products(path: Path, product_ids: Collection[str]) -> dict[str, Product]`
  - `load_questions(path: Path, product_ids: Collection[str]) -> dict[str, list[Question]]`
  - `file_sha256(path: Path) -> str`

规则：
- `source_line` 是记录在文件中的行号，从 1 开始；`source_position` 是评论在该记录 `reviews` 列表中的下标。
- `normalize_text` 后为空的评论跳过，不占用序号。
- 同一 ASIN 出现在多条记录时：先保留第一条记录的评论（按原顺序），再追加后续记录中 `dedupe_key` 尚未出现过的评论；同一条记录内部的重复评论保留。`review_index` 按最终顺序从 0 连续编号；`category` 取第一条记录的。
- `helpful` 不是长度为 2 的整数列表时，记为 `(0, 0)`。
- 不在 `product_ids` 里的商品不返回；文件里找不到的商品不报错（由调用方检查）。
- `qid` 统一转成 `str`；`is_answerable` 转成 `bool`；同一商品下问题文本相同但 `qid` 不同的行全部保留。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_text.py
def test_normalize_unescapes_and_collapses(self):
    self.assertEqual(normalize_text("  It&#39;s\n\t&quot;ok&quot;  "), "It's \"ok\"")
    self.assertEqual(dedupe_key("Great  Product"), "great product")

# tests/test_amazonqa.py（用 write_jsonl 在 TemporaryDirectory 中造数据）
def test_merges_duplicate_records_without_duplicate_reviews(self):
    # 第 1 行 A: ["Great", "Bad"]；第 2 行 B: ["x"]；第 3 行 A: ["great ", "New one"]
    p = load_products(path, {"A"})["A"]
    self.assertEqual([r.text for r in p.reviews], ["Great", "Bad", "New one"])
    self.assertEqual([r.review_id for r in p.reviews], ["A-r00000", "A-r00001", "A-r00002"])
    self.assertEqual((p.reviews[2].source_line, p.reviews[2].source_position), (3, 1))

def test_skips_empty_reviews_without_consuming_index(self):  # ["", "   ", "Real"]
    r = load_products(path, {"A"})["A"].reviews
    self.assertEqual([(x.review_index, x.source_position, x.text) for x in r], [(0, 2, "Real")])

def test_keeps_raw_text_and_stores_unescaped_text(self):  # "It&#39;s  fine"
    r = load_products(path, {"A"})["A"].reviews[0]
    self.assertEqual((r.raw_text, r.text), ("It&#39;s  fine", "It's fine"))

def test_bad_helpful_defaults_to_zero(self):  # helpful 缺失 / [1] / [2, 5]
    self.assertEqual([(x.helpful_yes, x.helpful_total) for x in reviews], [(0, 0), (0, 0), (2, 5)])

def test_only_requested_products_are_loaded(self):
    self.assertEqual(set(load_products(path, {"A"})), {"A"})

def test_load_questions_keeps_labels_and_duplicate_texts(self):
    # 两行 asin=A，问题文本相同，qid 7（is_answerable 1）和 8（is_answerable 0）；一行 asin=B
    qs = load_questions(path, {"A"})
    self.assertEqual([(q.qid, q.is_answerable) for q in qs["A"]], [("7", True), ("8", False)])
    self.assertNotIn("B", qs)

def test_file_sha256_matches_hashlib(self):
    self.assertEqual(file_sha256(path), hashlib.sha256(path.read_bytes()).hexdigest())
```

- [ ] **Step 2:** 运行 `python -m unittest tests.test_text tests.test_amazonqa -v`，预期 FAIL。
- [ ] **Step 3: 实现 `reviewproof/text.py` 和 `reviewproof/ingest/amazonqa.py`。** 逐行流式读取；先 `json.loads`，再按 `asin` 过滤；`file_sha256` 按 1 MiB 分块读取。
- [ ] **Step 4:** 运行同一命令，预期 PASS。
- [ ] **Step 5: 提交** `feat: read and merge AmazonQA product records with stable review ids`

### Task 4: 抽样清单

**Files:**
- Create: `reviewproof/ingest/sampling.py`
- Create: `samples/manifest_v1.json`（由脚本生成）
- Test: `tests/test_sampling.py`

**Interfaces:**
- Consumes: `normalize_text`、`file_sha256`（任务 3）
- Produces:
  - `BANDS = ("large", "medium", "small")`；`band_for(review_count: int) -> str`：≥200 为 `large`，50–199 为 `medium`，1–49 为 `small`，小于 1 时抛 `ValueError`
  - `ProductStats`（frozen dataclass）：`product_id: str, category: str, review_count: int, question_count: int`
  - `scan_product_stats(products_path: Path, questions_path: Path) -> list[ProductStats]`：只返回两个文件里都有的商品。`review_count` 取该 ASIN 各条记录中非空评论数的最大值（只用于分层，真实条数以导入结果为准）；`question_count` 取 `val-qar.jsonl` 中该 ASIN 的行数。
  - `select_sample(stats: Sequence[ProductStats], categories: Sequence[str], per_band: Mapping[str, int], min_questions: int, seed: int) -> tuple[list[ProductStats], list[dict]]`：返回选中的商品和缺额记录。
  - `assign_splits(selected: Sequence[ProductStats], dev_fraction: float, seed: int) -> dict[str, str]`：返回 `product_id → "dev" | "test"`。
  - `build_manifest(products_path: Path, questions_path: Path, selected: Sequence[ProductStats], splits: Mapping[str, str], shortfalls: list[dict], rules: Mapping) -> dict`
  - `main(argv: Sequence[str] | None = None) -> int`。参数：`--products`、`--questions`、`--output`；其余参数默认值为：categories `Electronics, Home_and_Kitchen, Automotive, Beauty`，per_band `large=10, medium=10, small=5`，min_questions `5`，seed `20261027`，dev_fraction `0.3`。

选取规则：
- 符合条件的商品：`category ∈ categories`、`question_count ≥ min_questions`、`review_count ≥ 1`。
- 每个品类单独用 `random.Random(f"{seed}:{category}")`，依次处理 large → medium → small。本层需求 = `per_band[band]` + 上一层的缺额。候选按 `product_id` 排序后 `rng.sample(候选, min(需求, len(候选)))`。
- 本层没选够时，追加缺额记录 `{"category", "band", "requested", "selected"}`，并把缺额带到下一层。small 层之后仍有缺额，只记录不再补。

划分规则：
- 按 (category, band) 分组；组内按 `product_id` 排序后用 `random.Random(f"{seed}:{category}:{band}")` 打乱；前 `math.floor(n * dev_fraction + 0.5)` 个为 dev，其余为 test。

清单格式（键按字母序输出，缩进 2）：

```json
{"manifest_version": 1, "dataset_id": "amazonqa-val-v1",
 "source": {"products_file": "data/val-qar_products.jsonl", "products_sha256": "…",
            "questions_file": "data/val-qar.jsonl", "questions_sha256": "…"},
 "rules": {"categories": […], "per_band": {…}, "min_questions": 5, "seed": 20261027,
           "dev_fraction": 0.3, "bands": {"large": ">=200", "medium": "50-199", "small": "1-49"}},
 "products": [{"product_id", "category", "band", "review_count", "question_count", "split"}],
 "shortfalls": […]}
```

- [ ] **Step 1: 写失败测试**（用合成的 `ProductStats` 列表，`scan_product_stats` 用 `write_jsonl` 造文件）

```python
def test_band_boundaries(self):
    self.assertEqual([band_for(n) for n in (1, 49, 50, 199, 200)], ["small", "small", "medium", "medium", "large"])
    with self.assertRaises(ValueError): band_for(0)

def test_selects_per_band_and_is_deterministic(self):  # 每层各 30 个候选
    per_band = {"large": 10, "medium": 10, "small": 5}
    a, short = select_sample(stats, ["Cat"], per_band, 5, 1)
    self.assertEqual(Counter(band_for(s.review_count) for s in a), per_band)
    self.assertEqual(short, [])
    self.assertEqual(a, select_sample(stats, ["Cat"], per_band, 5, 1)[0])
    self.assertNotEqual(a, select_sample(stats, ["Cat"], per_band, 5, 2)[0])

def test_shortfall_carries_to_next_band(self):  # 只有 3 个 large，30 个 medium，30 个 small
    a, short = select_sample(stats, ["Cat"], {"large": 10, "medium": 10, "small": 5}, 5, 1)
    self.assertEqual(Counter(band_for(s.review_count) for s in a)["medium"], 17)
    self.assertEqual(short, [{"category": "Cat", "band": "large", "requested": 10, "selected": 3}])

def test_filters_category_min_questions_and_empty_products(self):
    # 其他品类、question_count=4、review_count=0 的商品都不能被选中

def test_assign_splits_stratified(self):  # 一组 10 个，一组 5 个
    splits = assign_splits(selected, 0.3, 1)
    # 10 个的组 dev=3，5 个的组 dev=2；结果可复现

def test_scan_takes_max_review_count_over_duplicate_records_and_counts_questions(self):
    # A 两条记录，非空评论数 2 和 5 → review_count 5；val-qar 中 A 有 6 行 → question_count 6
```

- [ ] **Step 2:** 运行 `python -m unittest tests.test_sampling -v`，预期 FAIL。
- [ ] **Step 3: 实现 `reviewproof/ingest/sampling.py`。**
- [ ] **Step 4:** 运行同一命令，预期 PASS。
- [ ] **Step 5: 生成真实清单。** 运行 `python -m reviewproof.ingest.sampling --products data/val-qar_products.jsonl --questions data/val-qar.jsonl --output samples/manifest_v1.json`。预期：`products` 共 100 条，每个品类 25 条；`products_sha256` 等于 Global Constraints 中的值；`shortfalls` 只可能出现在 Automotive 的 large 层（已知只有 21 件候选，应当不缺）。把每个品类、每层、dev/test 的数量，以及题目总数（各商品 `question_count` 之和）打印出来核对。
- [ ] **Step 6: 提交** `feat: add reproducible sampling manifest (100 products, 4 categories)`

### Task 5: 句级切分

**Files:**
- Create: `reviewproof/ingest/chunking.py`
- Test: `tests/test_chunking.py`

**Interfaces:**
- Consumes: `Review`（任务 3）
- Produces:
  - `Chunk`（frozen dataclass）：`chunk_id: str, review_id: str, product_id: str, chunk_index: int, text: str, start_offset: int, end_offset: int, token_estimate: int`
  - `MAX_CHARS = 1200`、`TARGET_CHARS = 800`
  - `estimate_tokens(text: str) -> int` → `math.ceil(len(text) / 4)`
  - `chunk_review(review: Review, max_chars: int = MAX_CHARS, target_chars: int = TARGET_CHARS) -> list[Chunk]`

算法（测试不能完全确定，所以写明）：
- `len(review.text) ≤ max_chars` 时，整条评论作为一块：`(0, len)`。
- 否则：
  1. 用正则 `[^.!?]+(?:[.!?]+|$)` 找出句子区间，并去掉每个区间首尾的空白。
  2. 长于 `max_chars` 的句子，在 `max_chars` 以内最后一个空白处切开；没有空白就在 `max_chars` 处硬切。
  3. 从第一个句子开始，贪心地加入后续句子，直到"块起点到当前句末的长度"超过 `target_chars`；每块至少包含一个句子。
  4. 下一块从上一块的最后一个句子开始（重叠一句）；上一块只有一个句子时，从下一个句子开始，保证每次都向前推进。
- 每块的 `text == review.text[start_offset:end_offset]`；`chunk_id = f"{review.review_id}-c{i:02d}"`。

- [ ] **Step 1: 写失败测试**

```python
def test_short_review_is_one_chunk(self):
    c = chunk_review(review("Works great."))
    self.assertEqual([(x.chunk_id, x.start_offset, x.end_offset) for x in c], [("A-r00000-c00", 0, 12)])

def test_long_review_chunks_are_bounded_contiguous_and_overlapping(self):
    text = " ".join(f"Sentence number {i} talks about the battery life." for i in range(40))
    c = chunk_review(review(text))
    self.assertGreater(len(c), 1)
    for x in c:
        self.assertLessEqual(len(x.text), MAX_CHARS)
        self.assertEqual(x.text, text[x.start_offset:x.end_offset])
    self.assertEqual((c[0].start_offset, c[-1].end_offset), (0, len(text)))
    for a, b in zip(c, c[1:]):
        self.assertLess(b.start_offset, a.end_offset)   # 重叠一句

def test_giant_sentence_without_punctuation_is_hard_split(self):
    text = "word " * 1000   # 约 5,000 字符，没有句号
    c = chunk_review(review(text.strip()))
    self.assertTrue(all(len(x.text) <= MAX_CHARS for x in c))
    self.assertEqual(c[-1].end_offset, len(text.strip()))

def test_estimate_tokens(self):
    self.assertEqual(estimate_tokens("abcd" * 3), 3)
    self.assertEqual(estimate_tokens("abcde"), 2)
```

- [ ] **Step 2:** 运行 `python -m unittest tests.test_chunking -v`，预期 FAIL。
- [ ] **Step 3: 按上面的算法实现。**
- [ ] **Step 4:** 运行同一命令，预期 PASS。
- [ ] **Step 5: 提交** `feat: add sentence-based review chunking with offsets`

### Task 6: 写入数据库

**Files:**
- Create: `reviewproof/ingest/loader.py`
- Test: `tests/test_loader.py`

**Interfaces:**
- Consumes: `connect`、`apply_schema`（任务 2）；`Product`、`Question`（任务 3）；清单格式（任务 4）；`chunk_review`（任务 5）
- Produces:
  - `LoadReport`（frozen dataclass）：`products: int, reviews: int, chunks: int, questions: int`
  - `load_dataset(conn, manifest: Mapping, products: Mapping[str, Product], questions: Mapping[str, Sequence[Question]]) -> LoadReport`

规则：
- 一个事务完成，结束时提交。
- `datasets` 写入一行：`dataset_id`、`source_file = manifest["source"]["products_file"]`、`source_sha256`、整份清单。
- 清单里的商品如果不在 `products` 中，抛 `ValueError`，消息列出缺失的 ID，且不写入任何数据。
- `products` 的 `split`、`band` 取自清单；`review_count`、`chunk_count`、`token_estimate`（各块之和）按实际导入结果计算。
- `reviews`、`chunks`、`questions` 都按主键 upsert，重复执行结果不变。
- 块的文本没变时，保留原来的 `embedding` 和 `embedding_model`；文本变了，两者都置为 NULL。
- 删除该商品下不在本次块集合里的旧块。
- `LoadReport` 统计的是清单中商品在库里的实际行数。

- [ ] **Step 1: 写失败测试**（继承 `DatabaseTestCase`；用内存中构造的 `Product`、`Question` 和一份小清单）

```python
def test_load_is_idempotent(self):
    first = load_dataset(self.conn, manifest, products, questions)
    second = load_dataset(self.conn, manifest, products, questions)
    self.assertEqual(first, second)
    self.assertEqual(self.conn.execute("SELECT count(*) FROM chunks").fetchone()[0], first.chunks)

def test_embedding_kept_when_text_unchanged_and_cleared_when_changed(self):
    load_dataset(...)
    self.conn.execute("UPDATE chunks SET embedding = %s, embedding_model = 'm'", (np.ones(EMBEDDING_DIM, dtype=np.float32),))
    load_dataset(...)                       # 文本不变 → 保留
    # 断言 embedding_model 仍为 'm'
    load_dataset(..., products_with_edited_review_text)   # 文本变了 → NULL
    # 断言被修改评论对应块的 embedding 和 embedding_model 都是 NULL

def test_missing_manifest_product_raises_and_writes_nothing(self):
    with self.assertRaisesRegex(ValueError, "B"):
        load_dataset(self.conn, manifest_with_A_and_B, {"A": product_a}, {})
    self.assertEqual(self.conn.execute("SELECT count(*) FROM datasets").fetchone()[0], 0)

def test_question_fields_never_reach_chunks(self):
    # 回答文本里放一个哨兵字符串 "ANSWER_SENTINEL_42"，问题文本里放 "QUESTION_SENTINEL_42"
    self.assertEqual(self.conn.execute(
        "SELECT count(*) FROM chunks WHERE text LIKE '%SENTINEL_42%'").fetchone()[0], 0)

def test_product_split_band_and_counts_stored(self):
    row = self.conn.execute("SELECT split, band, review_count, chunk_count FROM products WHERE product_id = 'A'").fetchone()
    self.assertEqual(row, ("dev", "small", 3, 3))
```

- [ ] **Step 2:** 带 `REVIEWPROOF_TEST_DATABASE_URL` 运行 `python -m unittest tests.test_loader -v`，预期 FAIL。
- [ ] **Step 3: 实现 `reviewproof/ingest/loader.py`。** 批量写入用 `cursor.executemany`；块的"保留或清空向量"用 `ON CONFLICT (chunk_id) DO UPDATE SET ... embedding = CASE WHEN chunks.text = EXCLUDED.text THEN chunks.embedding END`，`embedding_model` 同理。
- [ ] **Step 4:** 运行同一命令，预期 PASS。
- [ ] **Step 5: 提交** `feat: load products, reviews, chunks and questions into Postgres`

### Task 7: 向量化（可断点续跑）

**Files:**
- Create: `reviewproof/embedding.py`、`reviewproof/ingest/embed.py`
- Modify: `tests/support.py`（新增 `FakeEmbedder`）
- Test: `tests/test_embedding.py`、`tests/test_embed_chunks.py`

**Interfaces:**
- Consumes: `connect`、`EMBEDDING_DIM`（任务 2）；`load_dataset`（任务 6，测试中用来准备数据）
- Produces:
  - `Embedder`（Protocol）：属性 `model_name: str`、`dim: int`；方法 `embed_documents(texts: Sequence[str]) -> list[list[float]]`、`embed_query(text: str) -> list[float]`
  - `SentenceTransformerEmbedder(model_name: str, dim: int, batch_size: int)`：首次调用时才加载 `sentence_transformers.SentenceTransformer(model_name)`；`encode` 传 `normalize_embeddings=True`；输出维度不等于 `dim` 时抛 `ValueError`。
  - `embed_pending_chunks(conn, embedder: Embedder, batch_size: int, product_ids: Collection[str] | None = None) -> int`：处理 `embedding IS NULL OR embedding_model IS DISTINCT FROM embedder.model_name` 的块，按 `chunk_id` 排序分批；每批写入后立即提交；写入前检查每个向量长度等于 `embedder.dim` 且全部是有限数，不满足就抛 `ValueError`，该批不写入。返回本次写入的块数。
  - `tests.support.FakeEmbedder(model_name: str = "fake", dim: int = 1024, fail_on_call: int | None = None)`：按文本的 SHA-256 生成确定性单位向量；第 `fail_on_call` 次调用 `embed_documents` 时抛 `RuntimeError`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_embed_chunks.py（DatabaseTestCase，先用 load_dataset 写入 5 个块）
def test_embeds_pending_then_nothing_left(self):
    self.assertEqual(embed_pending_chunks(self.conn, FakeEmbedder(), batch_size=2), 5)
    self.assertEqual(embed_pending_chunks(self.conn, FakeEmbedder(), batch_size=2), 0)

def test_resumes_after_failure(self):
    with self.assertRaises(RuntimeError):
        embed_pending_chunks(self.conn, FakeEmbedder(fail_on_call=2), batch_size=2)
    self.assertEqual(count_embedded(self.conn), 2)          # 第一批已提交
    self.assertEqual(embed_pending_chunks(self.conn, FakeEmbedder(), batch_size=2), 3)

def test_wrong_dimension_rejected_without_writing(self):
    with self.assertRaises(ValueError):
        embed_pending_chunks(self.conn, FakeEmbedder(dim=8), batch_size=2)
    self.assertEqual(count_embedded(self.conn), 0)

def test_model_change_reembeds_everything(self):
    embed_pending_chunks(self.conn, FakeEmbedder("a"), 2)
    self.assertEqual(embed_pending_chunks(self.conn, FakeEmbedder("b"), 2), 5)

def test_product_filter(self):  # 只处理 product_ids 中的商品

# tests/test_embedding.py（离线，patch sentence_transformers.SentenceTransformer）
def test_passes_normalize_and_checks_dim(self):
    # 假模型的 encode 返回 shape (n, 1024) 的数组 → 正常；返回 (n, 8) → ValueError
    # 断言 encode 被调用时 normalize_embeddings=True，且模型只加载一次
```

- [ ] **Step 2:** 运行 `python -m unittest tests.test_embedding tests.test_embed_chunks -v`，预期 FAIL。
- [ ] **Step 3: 实现。** 写入向量时转成 `numpy.asarray(vec, dtype=numpy.float32)`（`register_vector` 只接受 numpy 数组）。
- [ ] **Step 4:** 运行同一命令，预期 PASS（不带数据库变量时 DB 测试跳过、离线测试通过）。
- [ ] **Step 5: 提交** `feat: add resumable chunk embedding with BGE-M3`

### Task 8: 导入命令与真实数据导入

**Files:**
- Create: `reviewproof/ingest/__main__.py`
- Modify: `CLAUDE.md`（Commands 和 Architecture 中加入新包与导入命令）
- Test: `tests/test_ingest_cli.py`

**Interfaces:**
- Consumes: 任务 1–7 全部接口
- Produces:
  - `verify_ingest(conn, manifest: Mapping, require_embeddings: bool) -> dict`，返回 `{"products": int, "missing_products": list[str], "reviews": int, "chunks": int, "chunks_without_embedding": int, "questions": int, "ok": bool}`。`ok` 的条件：没有缺失商品，并且在 `require_embeddings` 为真时 `chunks_without_embedding == 0`。
  - `main(argv: Sequence[str] | None = None, embedder_factory: Callable[[Settings], Embedder] | None = None) -> int`。参数：`--manifest`（必填）、`--products-file`、`--questions-file`（默认取清单里的路径）、`--skip-embeddings`。

流程：
1. 只有在 `python -m reviewproof.ingest` 入口才调用 `dotenv.load_dotenv()`。
2. `load_settings()`。
3. `settings.embedding_dim != EMBEDDING_DIM` 时，在 stderr 输出错误并返回 2。
4. 两个源文件的 SHA-256 与清单不一致时，在 stderr 输出"清单与本地文件不一致"、文件路径和期望/实际值，并返回 2（此时还没有连接数据库）。
5. `load_products` / `load_questions` → `connect` → `apply_schema` → `load_dataset` → 除非 `--skip-embeddings`，否则 `embed_pending_chunks`（范围限于清单中的商品）→ `verify_ingest`。
6. 把 `LoadReport`、写入的向量数和校验结果作为 JSON 打印到 stdout；`ok` 为真返回 0，否则返回 1。
7. `embedder_factory` 默认为 `lambda s: SentenceTransformerEmbedder(s.embedding_model, s.embedding_dim, s.embedding_batch_size)`。

- [ ] **Step 1: 写失败测试**

```python
def test_sha_mismatch_exits_2_before_touching_db(self):   # 离线测试，不需要数据库
    code = main(["--manifest", str(manifest_with_wrong_sha)], embedder_factory=lambda s: FakeEmbedder())
    self.assertEqual(code, 2)

class IngestCliDatabaseTests(DatabaseTestCase):
    def test_end_to_end_with_fake_embedder(self):
        # 用 write_jsonl 造商品文件和问题文件，按真实 sha 写清单；patch REVIEWPROOF_DATABASE_URL 为测试库
        code = main(["--manifest", str(manifest)], embedder_factory=lambda s: FakeEmbedder())
        self.assertEqual(code, 0)
        report = verify_ingest(self.conn, manifest_dict, require_embeddings=True)
        self.assertTrue(report["ok"])
        self.assertEqual(report["chunks_without_embedding"], 0)

    def test_skip_embeddings_still_ok(self):
        self.assertEqual(main([... , "--skip-embeddings"], ...), 0)
```

- [ ] **Step 2:** 运行 `python -m unittest tests.test_ingest_cli -v`，预期 FAIL。
- [ ] **Step 3: 实现 `reviewproof/ingest/__main__.py`。**
- [ ] **Step 4:** 带数据库变量运行全量测试，预期全部 PASS；不带变量运行，预期 DB 测试 skipped、其余 PASS。
- [ ] **Step 5: 真实导入。** 首次运行会从 Hugging Face 下载 `BAAI/bge-m3`（约 2.3 GB），下载前先征得用户同意。运行 `docker compose up -d db`，再运行 `python -m reviewproof.ingest --manifest samples/manifest_v1.json`。预期输出 `"ok": true`，`products` 为 100，`chunks_without_embedding` 为 0。记录总评论数、总块数、题目数和向量化耗时。再运行一次同一命令，预期写入的向量数为 0，各项计数不变。
- [ ] **Step 6: 更新 `CLAUDE.md`。** Commands 中加入 `docker compose up -d db`、测试库变量和导入命令；Architecture 中说明 `reviewproof/`（新链路，第 1 阶段：数据底座）与 `bridge/`（旧链路，第 4 阶段删除）的关系。
- [ ] **Step 7: 提交** `feat: add ingest command and load the 100-product sample`
