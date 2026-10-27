<script setup lang="ts">
import { computed } from "vue";
import type { Product } from "../api/types";

const props = defineProps<{ products: Product[]; source: string; category: string; productId: string }>();
const emit = defineEmits<{ "update:category": [value: string]; "update:productId": [value: string] }>();

const categories = computed(() => [...new Set(props.products.map((p) => p.category))].sort());
const visible = computed(() => props.products.filter((p) => !props.category || p.category === props.category));
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
          <option v-for="c in categories" :key="c" :value="c">{{ c }}</option>
        </select>
      </label>
      <label class="grow">
        商品
        <select data-test="product-select" :value="productId"
                @change="emit('update:productId', ($event.target as HTMLSelectElement).value)">
          <option value="" disabled>请选择商品</option>
          <option v-for="p in visible" :key="p.product_id" :value="p.product_id">
            {{ p.product_id }} · 评论 {{ p.review_count }} 条 · {{ strategyLabel(p) }}
          </option>
        </select>
      </label>
    </div>
    <p class="muted">数据来源：{{ source || "AmazonQA 验证集抽样" }}。数据里没有商品名称，只显示 ASIN。</p>
  </section>
</template>
