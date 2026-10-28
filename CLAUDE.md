# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ReviewProof (GitHub `xuyul6/ReviewProof`): pre-purchase, single-product review QA on AmazonQA data, with answerability (answered / partial / unanswerable) as the core metric. Built phase by phase per `docs/roadmap_cn.md`; phases 1 (data foundation), 2 (hybrid retrieval), 3 (QA workflow), 4 (API and Vue frontend) and 5a (silver-label evaluation, pilot) are done. Docs are in Chinese.

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
python -m reviewproof.workflow --qid 17020                          # answer with citations; needs ANTHROPIC_API_KEY
python -m reviewproof.workflow --product B00007E7C8 --question "音量可以调节吗？" --json
python -m reviewproof.api                                           # API on :8000 (+ frontend/dist at /); loads models at startup
python -m reviewproof.mcp_server                                    # read-only MCP tools (REVIEWPROOF_MCP_TRANSPORT, default stdio)
python -m reviewproof.eval --split dev --name pilot-dev             # retrieval eval on the whole split + workflow/B0/B1 on 50 sampled questions; real Claude calls
python -m reviewproof.eval --split test --name NAME --retrieval-only   # retrieval only, no Claude calls
```

Frontend (`frontend/`, Vue 3 + Vite + TypeScript; Node ≥ 20, nvm-managed locally):

```bash
cd frontend && npm ci
npm run dev          # :5173, proxies /v1 and /healthz to :8000
npm run build        # frontend/dist, served by the API at /
npm run typecheck && npm test   # vue-tsc + Vitest (jsdom)
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

### Phase 3: QA workflow (`reviewproof/workflow/`, `reviewproof/llm.py`)

`answer_question(conn, product_id, question, *, llm, retrieval, max_rewrites)` in `graph.py` runs a LangGraph `StateGraph` (the only orchestration path; no hand-written fallback): plan (LLM `QueryPlan`: in or out of scope, yes/no, English search query — Chinese questions are translated here; out of scope is only price, stock, shipping, promotions, questions to the seller, recommending other products and definitive medical/legal judgements — warranty, specs, fit, authenticity and comparisons go to retrieval, matching the gold guideline) → retrieve (phase 2, `strategy="auto"`) → assess (LLM `EvidenceAssessment`: answered / partial / unanswerable, supporting ids, missing parts, rewrite query) → optional rewrite + re-retrieve (at most `REVIEWPROOF_MAX_REWRITES`, default 1; hybrid products only; each rewrite may add up to `evidence_k` new chunks; stops when the query repeats or nothing new comes back) → generate (LLM `GeneratedAnswer`: claims with chunk ids, unknowns, verdict; skipped when unanswerable) → verify (pure: drops citations not among this run's candidates or of another product; a claim left without citations is removed; the assessment's status stands unless no claims are left or a yes/no verdict is `unknown`, then unanswerable; listed gaps come only from the assessment's `missing`, never from the generator's caveats) → finalize (renders fixed copy in the question's language). Nodes degrade internally: a failed plan uses a heuristic (`service_status=degraded`); a failed retrieval, assessment or generation sets `service_status=failed` with `status=None` and an answer that says it was a service error — never "the reviews don't confirm this". `ClaudeLLM` calls `claude-haiku-5-5` via `client.messages.parse(output_format=<Pydantic schema>, output_config={"effort": ...})` with `max_tokens=4096`, no sampling parameters; every failure (including a missing `ANTHROPIC_API_KEY`, which the SDK would raise as a bare `TypeError`) becomes `LLMError`. Prompts (`prompts.py`) contain only the question and the product's chunk ids and texts, plus a fixed `Answer language: Chinese|English` line (the fixed copy exists only in those two). Yes/no questions generate with `GeneratedYesNoAnswer` (fields in writing order: `verdict_question` — the first yes/no question asked, logged in the trace — claims, unknowns, then the required verdict); its `unknown` (the excerpts lean neither way) makes verify set the status to unanswerable, so `unknown` goes with unanswerable only. `QAResult` carries the trace, search queries, retrieval modes and token usage; persisting runs is phase 4.

### Phase 4: API, frontend, MCP (`reviewproof/api/`, `frontend/`, `reviewproof/mcp_server.py`, `reviewproof/runs.py`)

`create_app()` builds `Services` once in the lifespan (applies the schema, loads LLM client, BGE-M3 and the reranker, warms them up). Endpoints: `GET /healthz`, `GET /v1/products?category=&split=`, `GET /v1/products/{id}/questions` (qid + text only), `GET /v1/evidence/{review_id}`, `GET /v1/runs/{run_id}`, `POST /v1/qa` (SSE). `qa.py` runs the workflow (`stream_answer`, which yields each node's `TraceEntry` then the `QAResult`) in a worker thread under a process-wide lock — one QA run at a time, because `BM25Cache` is not thread-safe and CPU inference gains nothing from overlap. Events: `run_started`, one `progress` per node, then `answer_chunk`s (the verified answer split by line / 120 chars — not token streaming), `citations`, `done` (incl. `persisted`), or `error` for unexpected exceptions. Service failures still end with `done` and HTTP 200; only bad input is 4xx (404 product, 422 blank question). Every run is saved to `qa_runs` before the answer is sent, and the worker finishes and saves even if the client disconnects. The Vue app talks to the API only through relative paths (`frontend/src/api/client.ts`, `parseSSE` in `sse.ts`, state in `composables/useAsk.ts`); the badge logic in `components/badge.ts` puts service failure before every answer status. The MCP server exposes `list_products`, `search_reviews` and `get_review` (no QA tool), sharing `reviewproof/api/queries.py` with the API.

### Phase 5a: evaluation (`reviewproof/eval/`)

Silver-label evaluation, written to `reports/eval/<name>/` (`config.json`, `retrieval.jsonl`, `per_question.jsonl`, `metrics.json`, `report.md`). `dataset.py` loads a split's questions and samples them stratified by category × silver label (seed 20261027). `alignment.py` maps AmazonQA's review snippets onto chunks (whitespace-insensitive, located in the full review then mapped by offset; snippets under 10 non-space characters are ignored) to build silver qrels. Every question aligns to about 10 chunks because AmazonQA's snippets are its own BM25 top 10, so retrieval scores measure agreement with that BM25, not relevance, and Recall@5 is capped near 0.5; the report says so. `retrieval_eval.py` ranks each question once and scores four variants (bm25, dense, hybrid, hybrid_rerank); full-context products, silver-unanswerable questions and unaligned questions are excluded and counted, and a hybrid variant is null when a channel it needs failed. `workflow_eval.py` runs the workflow, B0 (no evidence, `baselines.py`) and B1 (full context, hybrid-strategy products only), appending one JSON line per (question, system); rerunning the same command skips finished rows and retries rows that ended in a service failure or error. Cost prices each call at its own tier (`QAResult.usage`). `metrics.py` has rank metrics, bootstrap 95% intervals (1,000 resamples, seed 20261027) and paired differences; service failures are left out of every answerability denominator and counted separately. `report.py` compares B1 with the workflow only on B1's questions. Gold labels are phase 5b.

## Conventions

- New work follows the current phase plan in `docs/superpowers/plans/`; `docs/roadmap_cn.md` fixes the technical decisions (BGE-M3 1024-dim, RRF + bge-reranker-v2-m3, Claude Haiku 5.5 `claude-haiku-5-5` for generation via the official `anthropic` SDK, LangGraph as the only main path, exact per-product vector search without ANN indexes).
- System review IDs are generated, not Amazon IDs; never present them as Amazon review IDs. They are tied to the manifest's source file SHA, so labels and qrels must also record `dataset_id`.
- Questions, community answers, snippets and answerability labels live only in `questions` and must never reach `reviews`/`chunks` or any prompt.
- Env vars use the `REVIEWPROOF_` prefix; the Claude key is `ANTHROPIC_API_KEY`. `.env` is loaded only at command entry points (`reviewproof/ingest/__main__.py`), never on import.
- Product scope is set by `docs/product_positioning_cn.md` (includes non-goals). Docs are deliberately conservative about claims (e.g. silver vs. gold labels, system vs. Amazon IDs, heuristic hit rate vs. Recall@5); keep that precision.
