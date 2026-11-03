// Test helpers: a fake ApiClient whose answers each test controls.
import type { ApiClient } from "./api/client";
import type { DoneEvent, Evidence, QAEvent } from "./api/types";

export function doneEvent(overrides: Partial<DoneEvent> = {}): DoneEvent {
  return {
    run_id: "r1", status: "answered", service_status: "ok", verdict: "yes", route: "review_question",
    unknowns: [], search_queries: ["loud"], retrieval_modes: ["full_context"], rewrites_used: 0, llm_calls: 3,
    input_tokens: 100, output_tokens: 20, latency_ms: 1234, note: null, persisted: true, ...overrides,
  };
}

export function answeredEvents(text = "Verdict: Yes\nIt is too loud. [1]", done: Partial<DoneEvent> = {}): QAEvent[] {
  return [
    { event: "run_started", data: { run_id: "r1", product_id: "B000TEST01", question: "Is it loud?" } },
    { event: "progress", data: { node: "plan", status: "ok", detail: "route=review_question", latency_ms: 10 } },
    { event: "progress", data: { node: "retrieve", status: "ok", detail: "full_context", latency_ms: 5 } },
    { event: "answer_chunk", data: { text } },
    { event: "citations", data: { citations: [
      { number: 1, chunk_id: "B000TEST01-r00001-c00", review_id: "B000TEST01-r00001", text: "Too loud for me." }] } },
    { event: "done", data: doneEvent(done) },
  ];
}

export function fakeApi(overrides: Partial<Record<keyof ApiClient, unknown>> = {}, events: QAEvent[] = answeredEvents()) {
  const evidence: Evidence = {
    review_id: "B000TEST01-r00001", product_id: "B000TEST01", text: "Honestly? Too loud for me. Returned it.",
    helpful_yes: 1, helpful_total: 2, chunks: [{ chunk_id: "B000TEST01-r00001-c00", start_offset: 10, end_offset: 26 }],
  };
  return {
    health: async () => ({ database: true, llm_model: "claude-haiku-5-5", anthropic_key_set: true }),
    products: async () => ({
      source: "AmazonQA validation sample (amazonqa-val-v1)",
      products: [
        { product_id: "B000TEST01", category: "Electronics", title: null, review_count: 31, chunk_count: 31,
          split: "dev", strategy: "full_context" },
        { product_id: "B000TEST02", category: "Beauty", title: null, review_count: 900, chunk_count: 950,
          split: "test", strategy: "hybrid" },
      ],
    }),
    questions: async (productId: string) => ({ product_id: productId,
      questions: [{ qid: "q1", question_text: "Is it loud?" }] }),
    evidence: async () => evidence,
    run: async () => ({}),
    ask: async function* () { for (const e of events) yield e; },
    ...overrides,
  } as unknown as ApiClient;
}
