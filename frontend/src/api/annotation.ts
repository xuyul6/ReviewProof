// Client for the local annotation review API (served only with REVIEWPROOF_ANNOTATION=1).
import { toApiError } from "./client";

export type Answerability = "answered" | "partial" | "unanswerable";
export type Verdict = "yes" | "no" | "mixed" | "unknown";
export type Grade = 0 | 1 | 2;
export type ItemStatus = "pending" | "drafted" | "done";

export interface ItemSummary {
  qid: string;
  split: string;
  blind: boolean;
  status: ItemStatus;
}

export interface PoolChunk {
  chunk_id: string;
  review_id: string;
  text: string;
}

export interface LabelView {
  qid: string;
  source: "draft" | "blind" | "final";
  answerability: Answerability;
  verdict: Verdict | null;
  grades: Record<string, Grade>;
  rationale: string;
  annotator: string;
}

export interface AnnotationItem {
  qid: string;
  product_id: string;
  category: string;
  split: string;
  question_text: string;
  question_type: string;
  blind: boolean;
  pool: PoolChunk[];
  missing_chunks: string[]; // pool chunk ids no longer in the database
  extra_chunks: PoolChunk[];
  labels: Partial<Record<LabelView["source"], LabelView>>;
  draft_hidden: boolean;
  community_answers: string[];
}

export interface FinalBody {
  answerability: Answerability;
  verdict: Verdict | null;
  grades: Record<string, Grade>;
  rationale: string;
}

export function createAnnotationClient(fetchImpl: typeof fetch = (...args) => fetch(...args)) {
  const base = (set: string) => `/v1/annotation/${encodeURIComponent(set)}`;

  async function send<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetchImpl(path, init);
    if (!response.ok) throw await toApiError(response);
    return (await response.json()) as T;
  }

  const body = (method: string, payload: unknown): RequestInit =>
    ({ method, headers: { "content-type": "application/json" }, body: JSON.stringify(payload) });

  return {
    items: (set: string) => send<{ items: ItemSummary[] }>(`${base(set)}/items`).then((r) => r.items),
    item: (set: string, qid: string) => send<AnnotationItem>(`${base(set)}/items/${encodeURIComponent(qid)}`),
    blind: (set: string, qid: string, answerability: Answerability, verdict: Verdict | null) =>
      send<AnnotationItem>(`${base(set)}/items/${encodeURIComponent(qid)}/blind`,
        body("POST", { answerability, verdict })),
    final: (set: string, qid: string, payload: FinalBody) =>
      send<AnnotationItem>(`${base(set)}/items/${encodeURIComponent(qid)}/final`, body("PUT", payload)),
    search: (set: string, qid: string, q: string) =>
      send<{ chunks: PoolChunk[] }>(`${base(set)}/items/${encodeURIComponent(qid)}/search?${new URLSearchParams({ q })}`)
        .then((r) => r.chunks),
  };
}

export type AnnotationClient = ReturnType<typeof createAnnotationClient>;
