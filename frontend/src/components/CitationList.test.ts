import { flushPromises, mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import { fakeApi } from "../testing";
import CitationList from "./CitationList.vue";

const citations = [{ number: 1, chunk_id: "B000TEST01-r00001-c00", review_id: "B000TEST01-r00001",
  text: "Too loud for me." }];

describe("CitationList", () => {
  it("expanding loads the review and highlights the cited span", async () => {
    const wrapper = mount(CitationList, { props: { citations }, global: { provide: { api: fakeApi() } } });
    await wrapper.find('[data-test="citation-1"] button').trigger("click");
    await flushPromises();
    expect(wrapper.find("mark").text()).toBe("Too loud for me.");
    expect(wrapper.text()).toContain("Honestly?");
    expect(wrapper.text()).toContain("不是 Amazon 评论 ID");
  });

  it("missing review shows 原评论已不在当前数据中", async () => {
    const api = fakeApi({ evidence: async () => null });
    const wrapper = mount(CitationList, { props: { citations }, global: { provide: { api } } });
    await wrapper.find('[data-test="citation-1"] button').trigger("click");
    await flushPromises();
    expect(wrapper.text()).toContain("原评论已不在当前数据中");
  });

  it("opens the citation requested from outside", async () => {
    const wrapper = mount(CitationList, { props: { citations, open: 1 }, global: { provide: { api: fakeApi() } } });
    await flushPromises();
    expect(wrapper.find("mark").exists()).toBe(true);
  });
});
