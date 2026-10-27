import { flushPromises, mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import App from "./App.vue";
import { answeredEvents, fakeApi } from "./testing";

async function mountApp(api = fakeApi()) {
  const wrapper = mount(App, { global: { provide: { api } } });
  await flushPromises();
  return wrapper;
}

async function chooseProduct(wrapper: Awaited<ReturnType<typeof mountApp>>, productId = "B000TEST01") {
  await wrapper.find('[data-test="product-select"]').setValue(productId);
  await flushPromises();
}

async function ask(wrapper: Awaited<ReturnType<typeof mountApp>>, question = "Is it loud?") {
  await chooseProduct(wrapper);
  await wrapper.find('[data-test="question-input"]').setValue(question);
  await wrapper.find('[data-test="ask-button"]').trigger("click");
  await flushPromises();
}

describe("App", () => {
  it("renders products from the API", async () => {
    const wrapper = await mountApp();
    const options = wrapper.find('[data-test="product-select"]').text();
    expect(options).toContain("B000TEST01");
    expect(options).toContain("31");
    expect(wrapper.text()).toContain("AmazonQA");
  });

  it("filters products by category", async () => {
    const wrapper = await mountApp();
    await wrapper.find('[data-test="category-select"]').setValue("Beauty");
    expect(wrapper.find('[data-test="product-select"]').text()).not.toContain("B000TEST01");
    expect(wrapper.find('[data-test="product-select"]').text()).toContain("B000TEST02");
  });

  it('asking shows progress, the badge "能回答" and a citation [1]', async () => {
    const wrapper = await mountApp();
    await ask(wrapper);
    expect(wrapper.find('[data-test="progress"]').text()).toContain("规划");
    expect(wrapper.find('[data-test="badge"]').text()).toBe("能回答");
    expect(wrapper.find('[data-test="answer"]').text()).toContain("[1]");
    expect(wrapper.find('[data-test="citation-1"]').exists()).toBe(true);
  });

  it("service failure shows 服务出错 and never 评论无法确认", async () => {
    const events = answeredEvents("A service error stopped this run; this is not a finding about the reviews.",
      { service_status: "failed", status: null, verdict: null });
    const wrapper = await mountApp(fakeApi({}, events));
    await ask(wrapper);
    expect(wrapper.find('[data-test="badge"]').text()).toContain("服务出错");
    expect(wrapper.text()).not.toContain("评论无法确认");
  });

  it("out of scope shows 不在服务范围", async () => {
    const events = answeredEvents("This question is outside what reviews can answer: price",
      { route: "out_of_scope", status: "unanswerable", verdict: null });
    const wrapper = await mountApp(fakeApi({}, events));
    await ask(wrapper, "How much?");
    expect(wrapper.find('[data-test="badge"]').text()).toBe("不在服务范围");
  });

  it("unreachable API shows the start command", async () => {
    const wrapper = await mountApp(fakeApi({
      products: async () => { throw new TypeError("Failed to fetch"); },
      health: async () => { throw new TypeError("Failed to fetch"); },
    }));
    expect(wrapper.text()).toContain("python -m reviewproof.api");
  });

  it("asking is disabled until a product is chosen", async () => {
    const wrapper = await mountApp();
    await wrapper.find('[data-test="question-input"]').setValue("Is it loud?");
    expect(wrapper.find('[data-test="ask-button"]').attributes("disabled")).toBeDefined();
  });

  it("picking an example question fills the input", async () => {
    const wrapper = await mountApp();
    await chooseProduct(wrapper);
    await wrapper.find('[data-test="example-select"]').setValue("Is it loud?");
    expect((wrapper.find('[data-test="question-input"]').element as HTMLTextAreaElement).value).toBe("Is it loud?");
  });
});
