import { describe, expect, it } from "vitest";
import { ApiError, createApiClient } from "./client";

type Handler = (url: string, init?: RequestInit) => Response;

function fakeFetch(handler: Handler): typeof fetch {
  return (async (input: RequestInfo | URL, init?: RequestInit) => handler(String(input), init)) as typeof fetch;
}

function sseResponse(events: [string, unknown][]): Response {
  const text = events.map(([e, d]) => `event: ${e}\ndata: ${JSON.stringify(d)}\n\n`).join("");
  const bytes = new TextEncoder().encode(text);
  const body = new ReadableStream({
    start(controller) {
      controller.enqueue(bytes.slice(0, 25)); // split mid-event
      controller.enqueue(bytes.slice(25));
      controller.close();
    },
  });
  return new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } });
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

describe("createApiClient", () => {
  it("ask streams typed events", async () => {
    const seen: { url: string; init?: RequestInit }[] = [];
    const client = createApiClient(
      fakeFetch((url, init) => {
        seen.push({ url, init });
        return sseResponse([
          ["run_started", { run_id: "r1", product_id: "A", question: "q" }],
          ["progress", { node: "plan", status: "ok", detail: "", latency_ms: 1 }],
          ["answer_chunk", { text: "It is loud. [1]" }],
          ["citations", { citations: [] }],
          ["done", { run_id: "r1", status: "answered", service_status: "ok" }],
        ]);
      }),
    );
    const events = [];
    for await (const e of client.ask("A", "q")) events.push(e.event);
    expect(events).toEqual(["run_started", "progress", "answer_chunk", "citations", "done"]);
    expect(seen[0].url).toBe("/v1/qa");
    expect(seen[0].init?.method).toBe("POST");
    expect(JSON.parse(String(seen[0].init?.body))).toEqual({ product_id: "A", question: "q" });
  });

  it("ask throws ApiError on 404 before streaming", async () => {
    const client = createApiClient(fakeFetch(() => json({ detail: "product not found" }, 404)));
    const gen = client.ask("ZZZ", "q");
    await expect(gen.next()).rejects.toEqual(new ApiError(404, "product not found"));
  });

  it("evidence returns null on 404", async () => {
    const client = createApiClient(fakeFetch(() => json({ detail: "review not found in the current data" }, 404)));
    expect(await client.evidence("A-r09999")).toBeNull();
  });

  it("products passes category as a query parameter", async () => {
    const urls: string[] = [];
    const client = createApiClient(
      fakeFetch((url) => {
        urls.push(url);
        return json({ source: "s", products: [] });
      }),
    );
    await client.products("Beauty & Care");
    await client.products();
    expect(urls).toEqual(["/v1/products?category=Beauty+%26+Care", "/v1/products"]);
  });

  it("other errors throw ApiError with the detail", async () => {
    const client = createApiClient(fakeFetch(() => json({ detail: "run not found" }, 404)));
    await expect(client.run("nope")).rejects.toEqual(new ApiError(404, "run not found"));
  });
});
