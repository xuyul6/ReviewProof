import { describe, expect, it } from "vitest";
import { parseSSE } from "./sse";

async function* iter(chunks: string[]) {
  for (const c of chunks) yield c;
}
async function collect<T>(gen: AsyncIterable<T>): Promise<T[]> {
  const out: T[] = [];
  for await (const x of gen) out.push(x);
  return out;
}

describe("parseSSE", () => {
  it("reassembles events split across chunks", async () => {
    const chunks = ['event: progress\ndata: {"node":"pl', 'an"}\n\nevent: done\r\ndata: {"a":1}\r\n\r\n'];
    expect(await collect(parseSSE(iter(chunks)))).toEqual([
      { event: "progress", data: { node: "plan" } },
      { event: "done", data: { a: 1 } },
    ]);
  });

  it("joins multi-line data", async () => {
    expect(await collect(parseSSE(iter(['event: x\ndata: {"a":\ndata: 1}\n\n'])))).toEqual([
      { event: "x", data: { a: 1 } },
    ]);
  });

  it("defaults the event name to message", async () => {
    expect(await collect(parseSSE(iter(['data: {"b":2}\n\n'])))).toEqual([{ event: "message", data: { b: 2 } }]);
  });

  it("splits a separator that falls between chunks", async () => {
    const events = await collect(parseSSE(iter(['event: a\ndata: {}\n', '\nevent: b\ndata: {}\n\n'])));
    expect(events.map((e) => e.event)).toEqual(["a", "b"]);
  });
});
