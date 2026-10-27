import { parseSSE } from "./sse";
import type { Evidence, Health, ProductList, QAEvent, QuestionList, RunRecord } from "./types";

export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export async function toApiError(response: Response): Promise<ApiError> {
  let detail = response.statusText || `HTTP ${response.status}`;
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") detail = body.detail;
  } catch {
    // not JSON; keep the status text
  }
  return new ApiError(response.status, detail);
}

async function* readText(body: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      yield decoder.decode(value, { stream: true });
    }
    const rest = decoder.decode();
    if (rest) yield rest;
  } finally {
    reader.releaseLock();
  }
}

// All paths are relative: the Vite dev proxy or the same-origin FastAPI server answers them.
export function createApiClient(fetchImpl: typeof fetch = (...args) => fetch(...args)) {
  async function getJson<T>(path: string): Promise<T> {
    const response = await fetchImpl(path);
    if (!response.ok) throw await toApiError(response);
    return (await response.json()) as T;
  }

  return {
    health: () => getJson<Health>("/healthz"),

    products: (category?: string) =>
      getJson<ProductList>(category ? `/v1/products?${new URLSearchParams({ category })}` : "/v1/products"),

    questions: (productId: string) => getJson<QuestionList>(`/v1/products/${encodeURIComponent(productId)}/questions`),

    async evidence(reviewId: string): Promise<Evidence | null> {
      const response = await fetchImpl(`/v1/evidence/${encodeURIComponent(reviewId)}`);
      if (response.status === 404) return null;
      if (!response.ok) throw await toApiError(response);
      return (await response.json()) as Evidence;
    },

    run: (runId: string) => getJson<RunRecord>(`/v1/runs/${encodeURIComponent(runId)}`),

    async *ask(productId: string, question: string, signal?: AbortSignal): AsyncGenerator<QAEvent> {
      const response = await fetchImpl("/v1/qa", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ product_id: productId, question }),
        signal,
      });
      if (!response.ok || !response.body) throw await toApiError(response);
      for await (const message of parseSSE(readText(response.body))) yield message as QAEvent;
    },
  };
}

export type ApiClient = ReturnType<typeof createApiClient>;
