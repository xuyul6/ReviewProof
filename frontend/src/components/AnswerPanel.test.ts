import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import { doneEvent } from "../testing";
import AnswerPanel from "./AnswerPanel.vue";
import { statusBadge } from "./badge";

describe("statusBadge", () => {
  it.each([
    [{ service_status: "failed" as const, status: null, route: "review_question" as const },
      "服务出错（不代表评论里没有答案）", "error"],
    [{ route: "out_of_scope" as const, status: "unanswerable" as const }, "不在服务范围", "scope"],
    [{ status: "answered" as const }, "能回答", "ok"],
    [{ status: "partial" as const }, "部分能回答", "partial"],
    [{ status: "unanswerable" as const }, "评论无法确认", "none"],
  ])("%o → %s", (overrides, label, tone) => {
    expect(statusBadge(doneEvent(overrides))).toEqual({ label, tone });
  });
});

describe("AnswerPanel", () => {
  it("shows the degraded tag and clickable citation numbers", async () => {
    const wrapper = mount(AnswerPanel, {
      props: { done: doneEvent({ service_status: "degraded" }), answerText: "Loud. [1][2]\nSecond line" },
    });
    expect(wrapper.text()).toContain("部分检索通道降级");
    const refs = wrapper.findAll('[data-test="cite-ref"]');
    expect(refs.map((r) => r.text())).toEqual(["[1]", "[2]"]);
    await refs[1].trigger("click");
    expect(wrapper.emitted("cite")).toEqual([[2]]);
  });
});
