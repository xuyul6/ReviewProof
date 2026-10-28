-- ReviewProof schema. Idempotent: apply_schema() runs this on every ingest.
-- Questions, community answers and answerability labels live only in `questions`
-- (evaluation data); they never enter `reviews` or `chunks`.
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

CREATE TABLE IF NOT EXISTS qa_runs (
  run_id TEXT PRIMARY KEY, product_id TEXT NOT NULL REFERENCES products,
  question TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  route TEXT NOT NULL, status TEXT, service_status TEXT NOT NULL, verdict TEXT,
  llm_model TEXT NOT NULL, latency_ms DOUBLE PRECISION NOT NULL, result JSONB NOT NULL);
CREATE INDEX IF NOT EXISTS qa_runs_product_idx ON qa_runs (product_id, created_at DESC);

-- Gold annotation (phase 5b). qid references questions without ON DELETE CASCADE, so a re-ingest
-- that would drop an annotated question fails instead of silently deleting human labels.
CREATE TABLE IF NOT EXISTS annotation_items (
  label_set TEXT NOT NULL, qid TEXT NOT NULL REFERENCES questions, dataset_id TEXT NOT NULL,
  split TEXT NOT NULL, blind BOOLEAN NOT NULL, pool JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(), PRIMARY KEY (label_set, qid));
CREATE TABLE IF NOT EXISTS annotation_labels (
  label_set TEXT NOT NULL, qid TEXT NOT NULL,
  source TEXT NOT NULL CHECK (source IN ('draft', 'blind', 'final')),
  answerability TEXT NOT NULL, verdict TEXT, grades JSONB NOT NULL, rationale TEXT NOT NULL,
  annotator TEXT NOT NULL, guideline_version TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (label_set, qid, source), FOREIGN KEY (label_set, qid) REFERENCES annotation_items);
