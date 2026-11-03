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

describe("AnswerPanel note", () => {
  const note = { kind: "suggestion" as const, heading: "以下不是评论内容，是可以去哪里确认的建议：", text: "可以问卖家。" };

  it("renders the note below the answer", () => {
    const wrapper = mount(AnswerPanel, {
      props: { done: doneEvent({ status: "unanswerable", verdict: null, note }), answerText: "评论无法确认这个问题。" },
    });
    expect(wrapper.get('[data-test="note-heading"]').text()).toBe(note.heading);
    expect(wrapper.get('[data-test="note-text"]').text()).toBe(note.text);
    const html = wrapper.html();
    expect(html.indexOf('data-test="answer"')).toBeLessThan(html.indexOf('data-test="note"'));
  });

  it("shows no note when there is none", () => {
    const wrapper = mount(AnswerPanel, { props: { done: doneEvent(), answerText: "Loud. [1]" } });
    expect(wrapper.find('[data-test="note"]').exists()).toBe(false);
  });

  it("keeps note text plain, without citation buttons", () => {
    const wrapper = mount(AnswerPanel, {
      props: { done: doneEvent({ status: "unanswerable", note: { ...note, text: "见说明书 [1]" } }), answerText: "x" },
    });
    expect(wrapper.get('[data-test="note"]').findAll('[data-test="cite-ref"]')).toHaveLength(0);
    expect(wrapper.get('[data-test="note-text"]').text()).toBe("见说明书 [1]");
  });

  it("keeps the badge on the review status", () => {
    const wrapper = mount(AnswerPanel, {
      props: { done: doneEvent({ status: "unanswerable", note }), answerText: "评论无法确认这个问题。" },
    });
    expect(wrapper.get('[data-test="badge"]').text()).toBe("评论无法确认");
  });
});
