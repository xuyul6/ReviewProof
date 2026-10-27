// Mirrors the JSON returned by the ReviewProof API (reviewproof/api/app.py, reviewproof/api/qa.py).

export type Strategy = "full_context" | "hybrid";
export type AnswerStatus = "answered" | "partial" | "unanswerable";
export type ServiceStatus = "ok" | "degraded" | "failed";
export type Verdict = "yes" | "no" | "mixed" | "unknown";
export type Route = "review_question" | "out_of_scope";

export interface Health {
  database: boolean;
  llm_model: string;
  anthropic_key_set: boolean;
}

export interface Product {
  product_id: string;
  category: string;
  title: string | null;
  review_count: number;
  chunk_count: number;
  split: "dev" | "test";
  strategy: Strategy;
}

export interface ProductList {
  source: string;
  products: Product[];
}

export interface QuestionItem {
  qid: string;
  question_text: string;
}

export interface QuestionList {
  product_id: string;
  questions: QuestionItem[];
}

export interface EvidenceChunk {
  chunk_id: string;
  start_offset: number;
  end_offset: number;
}

export interface Evidence {
  review_id: string; // system-generated, not an Amazon review ID
  product_id: string;
  text: string;
  helpful_yes: number;
  helpful_total: number;
  chunks: EvidenceChunk[];
}

export interface Citation {
  number: number;
  chunk_id: string;
  review_id: string;
  text: string;
}

export interface TraceEvent {
  node: string;
  status: "ok" | "degraded" | "failed" | "skipped";
  detail: string;
  latency_ms: number;
}

export interface DoneEvent {
  run_id: string;
  status: AnswerStatus | null;
  service_status: ServiceStatus;
  verdict: Verdict | null;
  route: Route;
  unknowns: string[];
  search_queries: string[];
  retrieval_modes: string[];
  rewrites_used: number;
  llm_calls: number;
  input_tokens: number;
  output_tokens: number;
  latency_ms: number;
  persisted: boolean;
}

export interface SnapshotCandidate {
  chunk_id: string;
  review_id: string;
  dense_rank: number | null;
  dense_score: number | null;
  bm25_rank: number | null;
  bm25_score: number | null;
  rrf_score: number | null;
  rerank_score: number | null;
}

export interface RetrievalSnapshot {
  query: string;
  strategy: string;
  mode: string;
  service_status: ServiceStatus;
  channels: { name: string; status: string; reason: string | null; latency_ms: number; hits: number }[];
  candidates: SnapshotCandidate[];
}

export interface RunRecord {
  run_id: string;
  created_at: string;
  llm_model: string;
  product_id: string;
  question: string;
  answer_text: string;
  status: AnswerStatus | null;
  service_status: ServiceStatus;
  trace: TraceEvent[];
  retrievals: RetrievalSnapshot[];
  [key: string]: unknown;
}

export type QAEvent =
  | { event: "run_started"; data: { run_id: string; product_id: string; question: string } }
  | { event: "progress"; data: TraceEvent }
  | { event: "answer_chunk"; data: { text: string } }
  | { event: "citations"; data: { citations: Citation[] } }
  | { event: "done"; data: DoneEvent }
  | { event: "error"; data: { run_id: string; message: string } };
