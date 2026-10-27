<script setup lang="ts">
import type { Grade, PoolChunk } from "../api/annotation";

defineProps<{ chunk: PoolChunk; grade?: Grade; outside?: boolean }>();
const emit = defineEmits<{ grade: [value: Grade] }>();
const grades: Grade[] = [0, 1, 2];
</script>

<template>
  <div class="chunk-grader">
    <div class="chunk-head">
      <code>{{ chunk.chunk_id }}</code>
      <span v-if="outside" class="muted">池外</span>
      <span v-if="grade !== undefined" class="grade-buttons">
        <button v-for="g in grades" :key="g" type="button" class="choice" :class="{ active: grade === g }"
                :data-test="`grade-${chunk.chunk_id}-${g}`" @click="emit('grade', g)">{{ g }}</button>
      </span>
    </div>
    <p class="chunk-text">{{ chunk.text }}</p>
  </div>
</template>
