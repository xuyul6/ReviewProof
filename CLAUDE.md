# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ReviewProof (GitHub `xuyul6/ReviewProof`): pre-purchase, single-product review QA on AmazonQA data, with answerability (answered / partial / unanswerable) as the core metric. Built phase by phase per `docs/roadmap_cn.md`; phase 1 (data foundation) and phase 2 (hybrid retrieval) are done. Docs are in Chinese.

The old system (BRIDGE: `bridge/`, `app/`, Supabase + Gemini retrieval, the small-sample demo) lives in a separate repo, `xuyul6/BRIDGE`, checked out at `/Users/lux/llmproj/BRIDGE-by-EE6405-D31`. It is reference material only (roadmap section 4 says what to consult in which phase); never copy it in wholesale or add code to it.

## Commands

Python 3.10+ (local `.venv` is 3.13, CI uses 3.12). No linter/formatter is configured. Local Postgres 16 + pgvector via Docker on port 54329.

```bash
pip install -r requirements.txt
docker compose up -d db                                            # creates databases reviewproof and reviewproof_test
python -m reviewproof.ingest.sampling --products data/val-qar_products.jsonl --questions data/val-qar.jsonl --output samples/manifest_v1.json
python -m reviewproof.ingest --manifest samples/manifest_v1.json   # load + embed (BAAI/bge-m3) + verify; safe to re-run
python -m reviewproof.ingest --manifest samples/manifest_v1.json --skip-embeddings
python -m reviewproof.retrieval --qid 17020                         # rank one question's product; or --product ASIN --query TEXT
python -m reviewproof.retrieval --product B00007E7C8 --query "volume control?" --strategy hybrid --no-rerank --json
```

`data/` is git-ignored; download `val-qar_products.jsonl` and `val-qar.jsonl` from AmazonQA's S3 bucket. The ingest command refuses to run (exit 2) if the files' SHA-256 differ from the manifest. It prints embedding progress on stderr and the final JSON report on stdout. Embedding runs on CUDA if available, otherwise CPU (about 11 chunks/s on an M-series Mac); Apple MPS is not used by default because its per-batch-shape graph cache grows until it runs out of memory. `REVIEWPROOF_EMBEDDING_DEVICE` overrides the device. A full embedding of the 24,599 sample chunks takes roughly 40 minutes on CPU.

Tests use `unittest` (not pytest). Pure-logic tests run offline; database tests subclass `tests.support.DatabaseTestCase`, are skipped unless `REVIEWPROOF_TEST_DATABASE_URL` is set, and refuse any database whose name does not end in `_test` (they truncate every table). Test helpers (`DatabaseTestCase`, `FakeEmbedder`, `write_jsonl`, `make_product`, `make_manifest`) live in `tests/support.py`.

```bash
python -m unittest discover -s tests -b
REVIEWPROOF_TEST_DATABASE_URL=postgresql://reviewproof:reviewproof@localhost:54329/reviewproof_test python -m unittest discover -s tests -b
python -m unittest tests.test_chunking.ChunkingTests.test_short_review_is_one_chunk
```

## Architecture

### Phase 1: offline ingest

Everything is driven by `samples/manifest_v1.json` (100 AmazonQA products: 4 categories × review-count bands large ≥200 / medium 50–199 / small <50, dev/test split by product):
`ingest/sampling.py` builds the manifest → `ingest/amazonqa.py` reads `val-qar_products.jsonl` (full reviews; merges records that repeat an ASIN; skips empty reviews; generates system review IDs `{asin}-r{index:05d}`) and `val-qar.jsonl` (questions, `qid`, silver `is_answerable`) → `ingest/chunking.py` (reviews ≤1,200 chars stay whole, longer ones are packed by sentence to ~800 chars with one-sentence overlap; `chunk.text == review.text[start:end]`) → `ingest/loader.py` (upserts; a chunk keeps its embedding only while its text is unchanged; stale rows removed) → `ingest/embed.py` (resumable, per-batch commits) → `ingest/cli.py` verify.

Schema is `reviewproof/schema.sql` (datasets, products, reviews, chunks with `vector(1024)`, questions). `reviewproof/config.py` reads `REVIEWPROOF_*` settings. `reviewproof/text.py::normalize_text` (HTML-unescape + whitespace collapse) is the single normalization used for storage and dedupe.

### Phase 2: retrieval (`reviewproof/retrieval/`)

`retrieve(conn, product_id, query, *, embedder, embedding_model, reranker, bm25_cache, params, strategy)` in `retriever.py` returns a `RetrievalResult` (`types.py`). With `strategy="auto"`, products whose `token_estimate` ≤ `REVIEWPROOF_FULL_CONTEXT_MAX_TOKENS` (6,000; 28 of the 100 sample products) get all their chunks back as evidence (full context). Others go hybrid: `dense.py` (exact pgvector cosine sort within the product, no ANN index; refuses if any chunk lacks a vector from the current model) and `bm25.py` (`rank_bm25` per product, cached in memory, keyed by chunk count and text length; English `[a-z0-9]+` tokens, so Chinese queries skip BM25) each return top 50 → RRF (k=60) → top 40 form `pool` → `rerank.py` (`bge-reranker-v2-m3` CrossEncoder, `REVIEWPROOF_RERANK_MODEL`) → top 8 `candidates`. Every channel failure is caught and recorded in `channels` (status, reason, latency, hits); `mode` lists the channels that succeeded; `service_status` is `failed` when a recall channel failed and either no recall channel succeeded or the fused pool is empty (an empty result next to an outage says nothing about the reviews), `degraded` when some channel failed but candidates came back, otherwise `ok`; a channel that succeeds with 0 hits is not a failure. `failed` must never be read as "the reviews don't answer this". The evidence unit is the chunk (`chunk_id`) in both strategies. Channel latency includes lazy model loading on the first query.

## Conventions

- New work follows the current phase plan in `docs/superpowers/plans/`; `docs/roadmap_cn.md` fixes the technical decisions (BGE-M3 1024-dim, RRF + bge-reranker-v2-m3, Claude Haiku 5.5 `claude-haiku-5-5` for generation via the official `anthropic` SDK, LangGraph as the only main path, exact per-product vector search without ANN indexes).
- System review IDs are generated, not Amazon IDs; never present them as Amazon review IDs. They are tied to the manifest's source file SHA, so labels and qrels must also record `dataset_id`.
- Questions, community answers, snippets and answerability labels live only in `questions` and must never reach `reviews`/`chunks` or any prompt.
- Env vars use the `REVIEWPROOF_` prefix; the Claude key is `ANTHROPIC_API_KEY`. `.env` is loaded only at command entry points (`reviewproof/ingest/__main__.py`), never on import.
- Product scope is set by `docs/product_positioning_cn.md` (includes non-goals). Docs are deliberately conservative about claims (e.g. silver vs. gold labels, system vs. Amazon IDs, heuristic hit rate vs. Recall@5); keep that precision.
