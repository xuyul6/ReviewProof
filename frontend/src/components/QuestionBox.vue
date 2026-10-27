<script setup lang="ts">
import { ref, watch } from "vue";
import type { QuestionItem } from "../api/types";

const props = defineProps<{ questions: QuestionItem[]; running: boolean; disabled: boolean }>();
const emit = defineEmits<{ ask: [question: string]; cancel: [] }>();

const question = ref("");
const example = ref("");
watch(example, (value) => {
  if (value) question.value = value;
});
watch(() => props.questions, () => (example.value = ""));

function submit() {
  if (question.value.trim() && !props.running && !props.disabled) emit("ask", question.value.trim());
}
</script>

<template>
  <section class="card">
    <h2>提问</h2>
    <label v-if="questions.length">
      示例问题
      <select v-model="example" data-test="example-select">
        <option value="">（自己输入）</option>
        <option v-for="q in questions" :key="q.qid" :value="q.question_text">{{ q.question_text }}</option>
      </select>
    </label>
    <textarea v-model="question" data-test="question-input" rows="3" placeholder="例如：Is it loud? / 声音大吗？"
              @keydown.enter.exact.prevent="submit" />
    <div class="row">
      <button data-test="ask-button" :disabled="running || disabled || !question.trim()" @click="submit">提问</button>
      <button v-if="running" data-test="cancel-button" class="secondary" @click="emit('cancel')">取消</button>
    </div>
  </section>
</template>
