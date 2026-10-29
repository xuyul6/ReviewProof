<script setup lang="ts">
import { computed } from "vue";
import type { Product } from "../api/types";

const props = defineProps<{ products: Product[]; source: string; category: string; productId: string }>();
const emit = defineEmits<{ "update:category": [value: string]; "update:productId": [value: string] }>();

const CATEGORY_NAMES: Record<string, string> = {
  Automotive: "汽车用品", Beauty: "美妆个护", Electronics: "电子产品", Home_and_Kitchen: "家居厨房",
};
const categoryName = (c: string) => CATEGORY_NAMES[c] ?? c;
const TITLE_MAX = 48;

const categories = computed(() => [...new Set(props.products.map((p) => p.category))].sort());
const groups = computed(() =>
  categories.value
    .filter((c) => !props.category || c === props.category)
    .map((c) => ({ category: c, products: props.products.filter((p) => p.category === c) })),
);
const selected = computed(() => props.products.find((p) => p.product_id === props.productId));
const shortTitle = (p: Product) =>
  !p.title ? p.product_id : p.title.length > TITLE_MAX ? `${p.title.slice(0, TITLE_MAX).trimEnd()}…` : p.title;
const strategyLabel = (p: Product) => (p.strategy === "full_context" ? "全量上下文" : "混合检索");
</script>

<template>
  <section class="card">
    <h2>选择商品</h2>
    <div class="row">
      <label>
        品类
        <select data-test="category-select" :value="category"
                @change="emit('update:category', ($event.target as HTMLSelectElement).value)">
          <option value="">全部</option>
          <option v-for="c in categories" :key="c" :value="c">{{ categoryName(c) }}</option>
        </select>
      </label>
      <label class="grow">
        商品
        <select data-test="product-select" :value="productId"
                @change="emit('update:productId', ($event.target as HTMLSelectElement).value)">
          <option value="" disabled>请选择商品</option>
          <optgroup v-for="g in groups" :key="g.category" :label="categoryName(g.category)">
            <option v-for="p in g.products" :key="p.product_id" :value="p.product_id">
              {{ shortTitle(p) }} · 评论 {{ p.review_count }} 条
            </option>
          </optgroup>
        </select>
      </label>
    </div>
    <p v-if="selected" class="selected" data-test="selected-product">
      <strong>{{ selected.title ?? "（没有找到这件商品的标题）" }}</strong><br />
      <span class="muted">ASIN {{ selected.product_id }} · 评论 {{ selected.review_count }} 条 · {{ strategyLabel(selected) }}</span>
    </p>
    <p class="muted">
      数据来源：{{ source || "AmazonQA 验证集抽样" }}。商品标题按 ASIN 取自 Amazon Reviews 2023 元数据，可能和评论当时的商品页面不完全一致；没有标题的商品只显示 ASIN。
    </p>
  </section>
</template>

<style scoped>
.selected { margin: 0.75rem 0 0; }
</style>
