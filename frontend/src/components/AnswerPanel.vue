<script setup lang="ts">
import { computed } from "vue";
import type { DoneEvent } from "../api/types";
import { statusBadge } from "./badge";

const props = defineProps<{ done: DoneEvent | null; answerText: string }>();
const emit = defineEmits<{ cite: [number: number] }>();

const badge = computed(() => statusBadge(props.done));
// Split the text into plain parts and [n] citation markers.
const parts = computed(() =>
  props.answerText.split(/(\[\d+\])/).filter(Boolean).map((part) => {
    const match = /^\[(\d+)\]$/.exec(part);
    return match ? { cite: Number(match[1]), text: part } : { cite: null, text: part };
  }),
);
</script>

<template>
  <section class="card answer">
    <div class="badges">
      <span v-if="badge" data-test="badge" :class="['badge', `badge-${badge.tone}`]">{{ badge.label }}</span>
      <span v-if="done?.service_status === 'degraded'" class="badge badge-degraded">部分检索通道降级</span>
    </div>
    <p class="answer-text" data-test="answer">
      <template v-for="(part, i) in parts" :key="i">
        <button v-if="part.cite !== null" class="cite" data-test="cite-ref" @click="emit('cite', part.cite)">
          {{ part.text }}
        </button>
        <template v-else>{{ part.text }}</template>
      </template>
    </p>
  </section>
</template>
