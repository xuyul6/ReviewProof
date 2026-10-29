import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import type { Product } from "../api/types";
import ProductPicker from "./ProductPicker.vue";

const products: Product[] = [
  { product_id: "B000TEST01", category: "Electronics", title: "Koss PortaPro On-Ear Headphones with Comfort Zone Volume Control",
    review_count: 1962, chunk_count: 2000, split: "dev", strategy: "hybrid" },
  { product_id: "B000TEST02", category: "Home_and_Kitchen", title: "Copper Pepper Mill", review_count: 62, chunk_count: 62,
    split: "dev", strategy: "hybrid" },
  { product_id: "B000TEST03", category: "Electronics", title: null, review_count: 19, chunk_count: 19, split: "dev",
    strategy: "full_context" },
];

const mountPicker = (props: Partial<{ category: string; productId: string }> = {}) =>
  mount(ProductPicker, { props: { products, source: "AmazonQA", category: "", productId: "", ...props } });

describe("ProductPicker", () => {
  it("groups products by category under Chinese names and shows titles with review counts", () => {
    const select = mountPicker().find('[data-test="product-select"]');
    const groups = select.findAll("optgroup").map((g) => g.attributes("label"));
    expect(groups).toEqual(["电子产品", "家居厨房"]);
    const electronics = select.findAll("optgroup")[0].findAll("option").map((o) => o.text());
    expect(electronics[0]).toContain("Koss PortaPro");
    expect(electronics[0]).toContain("评论 1962 条");
    expect(electronics[0]).not.toContain("B000TEST01");
  });

  it("truncates long titles in the option", () => {
    const option = mountPicker().find('[data-test="product-select"] option[value="B000TEST01"]').text();
    expect(option).toContain("…");
    expect(option.length).toBeLessThan(70);
  });

  it("falls back to the ASIN when a product has no title", () => {
    const option = mountPicker().find('[data-test="product-select"] option[value="B000TEST03"]').text();
    expect(option).toContain("B000TEST03");
  });

  it("shows the selected product's full title, ASIN and retrieval mode", () => {
    const selected = mountPicker({ productId: "B000TEST01" }).find('[data-test="selected-product"]').text();
    expect(selected).toContain("Koss PortaPro On-Ear Headphones with Comfort Zone Volume Control");
    expect(selected).toContain("B000TEST01");
    expect(selected).toContain("混合检索");
  });

  it("category options use Chinese names", () => {
    const options = mountPicker().findAll('[data-test="category-select"] option').map((o) => o.text());
    expect(options).toEqual(["全部", "电子产品", "家居厨房"]);
  });
});
