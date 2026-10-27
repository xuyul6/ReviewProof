import { ref } from "vue";
import { ApiError, type ApiClient } from "../api/client";
import type { Citation, DoneEvent, TraceEvent } from "../api/types";

export type AskPhase = "idle" | "running" | "done" | "error";

// Reactive state for one question: progress arrives live, the answer arrives in verified chunks.
export function useAsk(client: ApiClient) {
  const phase = ref<AskPhase>("idle");
  const runId = ref<string | null>(null);
  const progress = ref<TraceEvent[]>([]);
  const answerText = ref("");
  const citations = ref<Citation[]>([]);
  const done = ref<DoneEvent | null>(null);
  const errorMessage = ref("");
  let controller: AbortController | null = null;

  function reset() {
    runId.value = null;
    progress.value = [];
    answerText.value = "";
    citations.value = [];
    done.value = null;
    errorMessage.value = "";
  }

  async function start(productId: string, question: string): Promise<void> {
    controller?.abort();
    const current = new AbortController();
    controller = current;
    reset();
    phase.value = "running";
    try {
      for await (const message of client.ask(productId, question, current.signal)) {
        if (current.signal.aborted) return;
        switch (message.event) {
          case "run_started":
            runId.value = message.data.run_id;
            break;
          case "progress":
            progress.value = [...progress.value, message.data];
            break;
          case "answer_chunk":
            answerText.value += message.data.text;
            break;
          case "citations":
            citations.value = message.data.citations;
            break;
          case "done":
            done.value = message.data;
            phase.value = "done";
            break;
          case "error":
            errorMessage.value = message.data.message;
            phase.value = "error";
            break;
        }
      }
      // The stream ended without done or error (server restart, dropped proxy connection).
      if (!current.signal.aborted && phase.value === "running") {
        errorMessage.value = "连接中断，未收到完整结果";
        phase.value = "error";
      }
    } catch (error) {
      if (current.signal.aborted) return;
      errorMessage.value = error instanceof ApiError ? error.detail : String(error);
      phase.value = "error";
    }
  }

  function cancel() {
    controller?.abort();
    controller = null;
    reset();
    phase.value = "idle";
  }

  return { phase, runId, progress, answerText, citations, done, errorMessage, start, cancel };
}
