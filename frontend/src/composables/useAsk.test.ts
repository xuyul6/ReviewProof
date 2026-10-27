import { describe, expect, it } from "vitest";
import { ApiError, type ApiClient } from "../api/client";
import type { QAEvent } from "../api/types";
import { useAsk } from "./useAsk";

function clientWith(events: QAEvent[] | (() => AsyncGenerator<QAEvent>), error?: Error): ApiClient {
  return {
    ask: async function* () {
      if (typeof events === "function") {
        yield* events();
        return;
      }
      for (const e of events) yield e;
      if (error) throw error;
    },
  } as unknown as ApiClient;
}

const done = (service_status: string, status: string | null) =>
  ({ event: "done", data: { run_id: "r1", status, service_status } }) as unknown as QAEvent;

describe("useAsk", () => {
  it("accumulates progress and answer chunks, then done", async () => {
    const ask = useAsk(
      clientWith([
        { event: "run_started", data: { run_id: "r1", product_id: "A", question: "q" } },
        { event: "progress", data: { node: "plan", status: "ok", detail: "", latency_ms: 1 } },
        { event: "answer_chunk", data: { text: "Line one\n" } },
        { event: "answer_chunk", data: { text: "Line two [1]" } },
        { event: "citations", data: { citations: [{ number: 1, chunk_id: "A-r00001-c00", review_id: "A-r00001", text: "t" }] } },
        done("ok", "answered"),
      ]),
    );
    await ask.start("A", "q");
    expect(ask.phase.value).toBe("done");
    expect(ask.progress.value.map((p) => p.node)).toEqual(["plan"]);
    expect(ask.answerText.value).toBe("Line one\nLine two [1]");
    expect(ask.citations.value).toHaveLength(1);
    expect(ask.done.value?.status).toBe("answered");
    expect(ask.runId.value).toBe("r1");
  });

  it("service failure finishes as done, not error", async () => {
    const ask = useAsk(clientWith([done("failed", null)]));
    await ask.start("A", "q");
    expect(ask.phase.value).toBe("done");
    expect(ask.done.value?.service_status).toBe("failed");
  });

  it("error event and thrown ApiError set phase error with message", async () => {
    const a = useAsk(clientWith([{ event: "error", data: { run_id: "r1", message: "RuntimeError: boom" } }]));
    await a.start("A", "q");
    expect([a.phase.value, a.errorMessage.value]).toEqual(["error", "RuntimeError: boom"]);
    const b = useAsk(clientWith([], new ApiError(404, "product not found")));
    await b.start("ZZZ", "q");
    expect([b.phase.value, b.errorMessage.value]).toEqual(["error", "product not found"]);
  });

  it("start resets the previous answer", async () => {
    const ask = useAsk(clientWith([{ event: "answer_chunk", data: { text: "old" } }, done("ok", "answered")]));
    await ask.start("A", "q");
    await ask.start("A", "q");
    expect(ask.answerText.value).toBe("old");
  });

  it("cancel aborts and returns to idle", async () => {
    let release: () => void = () => {};
    const gate = new Promise<void>((resolve) => (release = resolve));
    const ask = useAsk(
      clientWith(async function* () {
        yield { event: "progress", data: { node: "plan", status: "ok", detail: "", latency_ms: 1 } } as QAEvent;
        await gate;
        yield done("ok", "answered");
      }),
    );
    const running = ask.start("A", "q");
    await Promise.resolve();
    await Promise.resolve();
    expect(ask.phase.value).toBe("running");
    ask.cancel();
    release();
    await running;
    expect(ask.phase.value).toBe("idle");
    expect(ask.done.value).toBeNull();
  });
});
