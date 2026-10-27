import { flushPromises, mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";
import type { AnnotationClient, AnnotationItem, ItemSummary } from "../api/annotation";
import { ApiError } from "../api/client";
import AnnotateApp from "./AnnotateApp.vue";

const pool = [
  { chunk_id: "c1", review_id: "r1", text: "Too loud for me." },
  { chunk_id: "c2", review_id: "r2", text: "Shipped fast." },
];

function item(qid: string, overrides: Partial<AnnotationItem> = {}): AnnotationItem {
  return {
    qid, product_id: "A", category: "Electronics", split: "dev", question_text: `Is ${qid} loud?`,
    question_type: "yesno", blind: false, pool, extra_chunks: [], draft_hidden: false, community_answers: ["Yes."],
    labels: { draft: { qid, source: "draft", answerability: "answered", verdict: "yes", grades: { c1: 2, c2: 0 },
                       rationale: "DRAFT-R", annotator: "agent:t" } },
    ...overrides,
  };
}

function fakeClient(summaries: ItemSummary[], overrides: Partial<AnnotationClient> = {}) {
  const client = {
    items: vi.fn(async () => summaries),
    item: vi.fn(async (_set: string, qid: string) => item(qid)),
    blind: vi.fn(),
    final: vi.fn(async (_set: string, qid: string) => item(qid)),
    search: vi.fn(async () => []),
    ...overrides,
  };
  return client as unknown as AnnotationClient & typeof client;
}

async function mountWith(client: AnnotationClient) {
  const wrapper = mount(AnnotateApp, { props: { labelSet: "gold_v1" },
                                       global: { provide: { annotationApi: client } } });
  await flushPromises();
  return wrapper;
}

const drafted = (qid: string, blind = false): ItemSummary => ({ qid, split: "dev", blind, status: "drafted" });

describe("AnnotateApp", () => {
  it("hides the draft until the blind label is submitted", async () => {
    const hidden = item("q3", { blind: true, draft_hidden: true, labels: {} });
    const revealed = item("q3", { blind: true });
    const client = fakeClient([drafted("q3", true)], {
      item: vi.fn(async () => hidden),
      blind: vi.fn(async () => revealed),
    });
    const wrapper = await mountWith(client);
    expect(wrapper.text()).toContain("Is q3 loud?");
    expect(wrapper.text()).not.toContain("DRAFT-R");
    expect(wrapper.find('[data-test="save"]').exists()).toBe(false);
    await wrapper.find('[data-test="ans-partial"]').trigger("click");
    await wrapper.find('[data-test="verdict-mixed"]').trigger("click");
    await wrapper.find('[data-test="blind-submit"]').trigger("click");
    await flushPromises();
    expect(client.blind).toHaveBeenCalledWith("gold_v1", "q3", "partial", "mixed");
    expect((wrapper.find("textarea").element as HTMLTextAreaElement).value).toBe("DRAFT-R");
    expect(wrapper.find('[data-test="save"]').exists()).toBe(true);
  });

  it("prefills from the draft and saves changed grades and answerability", async () => {
    const client = fakeClient([drafted("q1")]);
    const wrapper = await mountWith(client);
    expect(wrapper.find('[data-test="ans-answered"]').classes()).toContain("active");
    await wrapper.find('[data-test="grade-c2-1"]').trigger("click");
    await wrapper.find('[data-test="ans-partial"]').trigger("click");
    await wrapper.find('[data-test="save"]').trigger("click");
    await flushPromises();
    expect(client.final).toHaveBeenCalledWith("gold_v1", "q1", { answerability: "partial", verdict: "yes",
                                                                  grades: { c1: 2, c2: 1 }, rationale: "DRAFT-R" });
  });

  it("shows the 422 detail and stays on the item", async () => {
    const client = fakeClient([drafted("q1"), drafted("q2")], {
      final: vi.fn(async () => { throw new ApiError(422, "answered needs at least one grade-2 chunk"); }),
    });
    const wrapper = await mountWith(client);
    await wrapper.find('[data-test="save"]').trigger("click");
    await flushPromises();
    expect(wrapper.find('[data-test="error"]').text()).toContain("grade-2");
    expect(wrapper.text()).toContain("Is q1 loud?");
  });

  it("next pending skips done items", async () => {
    const summaries = [drafted("q1"), { ...drafted("q2"), status: "done" as const }, drafted("q3")];
    const client = fakeClient(summaries);
    vi.mocked(client.items).mockImplementationOnce(async () => summaries)
      .mockImplementation(async () => [{ ...summaries[0], status: "done" as const }, summaries[1], summaries[2]]);
    const wrapper = await mountWith(client);
    await wrapper.find('[data-test="save"]').trigger("click");
    await flushPromises();
    expect(client.item).toHaveBeenLastCalledWith("gold_v1", "q3");
    expect(wrapper.text()).toContain("Is q3 loud?");
  });

  it("adds a searched out-of-pool chunk to the saved grades", async () => {
    const client = fakeClient([drafted("q1")], {
      search: vi.fn(async () => [{ chunk_id: "c9", review_id: "r9", text: "Whisper quiet at night." }]),
    });
    const wrapper = await mountWith(client);
    await wrapper.find('[data-test="search-input"]').setValue("quiet");
    await wrapper.find('[data-test="search-go"]').trigger("click");
    await flushPromises();
    await wrapper.find('[data-test="add-c9"]').trigger("click");
    await wrapper.find('[data-test="grade-c9-2"]').trigger("click");
    await wrapper.find('[data-test="save"]').trigger("click");
    await flushPromises();
    expect(client.search).toHaveBeenCalledWith("gold_v1", "q1", "quiet");
    expect(vi.mocked(client.final).mock.calls[0][2].grades).toEqual({ c1: 2, c2: 0, c9: 2 });
  });
});
