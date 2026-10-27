<script setup lang="ts">
import { computed, inject, onMounted, ref, watch } from "vue";
import type { ApiClient } from "./api/client";
import type { Health, Product, QuestionItem } from "./api/types";
import AnswerPanel from "./components/AnswerPanel.vue";
import CitationList from "./components/CitationList.vue";
import HealthBanner from "./components/HealthBanner.vue";
import ProductPicker from "./components/ProductPicker.vue";
import ProgressList from "./components/ProgressList.vue";
import QuestionBox from "./components/QuestionBox.vue";
import RunDetails from "./components/RunDetails.vue";
import { useAsk } from "./composables/useAsk";

const api = inject<ApiClient>("api")!;
const health = ref<Health | null>(null);
const unreachable = ref(false);
const products = ref<Product[]>([]);
const source = ref("");
const category = ref("");
const productId = ref("");
const questions = ref<QuestionItem[]>([]);
const openCitation = ref<number | null>(null);
const ask = useAsk(api);
const running = computed(() => ask.phase.value === "running");

onMounted(async () => {
  try {
    const [h, list] = await Promise.all([api.health(), api.products()]);
    health.value = h;
    products.value = list.products;
    source.value = list.source;
  } catch {
    unreachable.value = true;
  }
});

watch(category, () => {
  if (productId.value && !products.value.some((p) => p.product_id === productId.value && (!category.value || p.category === category.value))) {
    productId.value = "";
  }
});

watch(productId, async (id) => {
  ask.cancel();
  openCitation.value = null;
  questions.value = [];
  if (!id) return;
  try {
    questions.value = (await api.questions(id)).questions;
  } catch {
    questions.value = [];
  }
});

function onAsk(question: string) {
  openCitation.value = null;
  void ask.start(productId.value, question);
}
</script>

<template>
  <div class="page">
    <header>
      <h1>ReviewProof</h1>
      <p class="muted">购前单商品评论问答：只用这件商品的买家评论回答，每个结论附可回查的评论引用。</p>
    </header>

    <HealthBanner :health="health" :unreachable="unreachable" />

    <ProductPicker v-model:category="category" v-model:product-id="productId" :products="products" :source="source" />

    <QuestionBox :questions="questions" :running="running" :disabled="!productId" @ask="onAsk" @cancel="ask.cancel" />

    <section v-if="ask.phase.value !== 'idle'" class="card">
      <h2>进度</h2>
      <ProgressList :progress="ask.progress.value" :running="running" />
      <p v-if="ask.phase.value === 'error'" class="banner banner-error">出错了：{{ ask.errorMessage.value }}</p>
    </section>

    <template v-if="ask.done.value">
      <AnswerPanel :done="ask.done.value" :answer-text="ask.answerText.value" @cite="openCitation = $event" />
      <CitationList :citations="ask.citations.value" :open="openCitation" />
      <RunDetails :done="ask.done.value" :progress="ask.progress.value" />
    </template>

    <footer class="muted">
      进度实时显示；答案在引用核验完成后分段显示，不是逐字生成。评论 ID 由系统生成，不是 Amazon 评论 ID。
    </footer>
  </div>
</template>
