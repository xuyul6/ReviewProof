import { flushPromises, mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";
import { doneEvent, fakeApi } from "../testing";
import RunDetails from "./RunDetails.vue";

const record = {
  run_id: "r1",
  retrievals: [{
    query: "loud", strategy: "hybrid", mode: "dense+bm25+rerank", service_status: "ok", channels: [],
    candidates: [{ chunk_id: "B000TEST01-r00001-c00", review_id: "B000TEST01-r00001", dense_rank: 1, dense_score: 0.9,
                   bm25_rank: 2, bm25_score: 3.1, rrf_score: 0.0325, rerank_score: 0.81 }],
  }],
};

async function open(api: ReturnType<typeof fakeApi>) {
  const wrapper = mount(RunDetails, { props: { done: doneEvent(), progress: [] }, global: { provide: { api } } });
  const details = wrapper.find("details");
  (details.element as HTMLDetailsElement).open = true;
  await details.trigger("toggle");
  await flushPromises();
  return wrapper;
}

describe("RunDetails", () => {
  it("loads the run once and shows the candidate table", async () => {
    const run = vi.fn(async () => record);
    const wrapper = await open(fakeApi({ run }));
    expect(wrapper.text()).toContain("B000TEST01-r00001-c00");
    expect(wrapper.text()).toContain("0.810");
    await wrapper.find("details").trigger("toggle");
    expect(run).toHaveBeenCalledTimes(1);
  });

  it("shows a message when the run cannot be fetched", async () => {
    const wrapper = await open(fakeApi({ run: async () => { throw new Error("boom"); } }));
    expect(wrapper.text()).toContain("运行记录暂时取不到");
  });
});
