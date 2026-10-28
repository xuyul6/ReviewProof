import { describe, expect, it } from "vitest";
import { ApiError } from "./client";
import { createAnnotationClient } from "./annotation";

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

describe("createAnnotationClient", () => {
  it("puts the final label as JSON on the item path", async () => {
    const seen: { url: string; init?: RequestInit }[] = [];
    const client = createAnnotationClient((async (url: RequestInfo | URL, init?: RequestInit) => {
      seen.push({ url: String(url), init });
      return json({ qid: "q 1" });
    }) as typeof fetch);
    await client.final("gold_v1", "q 1", { answerability: "answered", verdict: null, grades: { c: 2 }, rationale: "r" });
    expect(seen[0].url).toBe("/v1/annotation/gold_v1/items/q%201/final");
    expect(seen[0].init?.method).toBe("PUT");
    expect(JSON.parse(String(seen[0].init?.body))).toEqual({ answerability: "answered", verdict: null,
      grades: { c: 2 }, rationale: "r" });
  });

  it("raises ApiError with the server detail", async () => {
    const client = createAnnotationClient((async () => json({ detail: "needs grade-2" }, 422)) as typeof fetch);
    await expect(client.item("gold_v1", "q1")).rejects.toEqual(new ApiError(422, "needs grade-2"));
  });
});
