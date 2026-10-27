<script setup lang="ts">
import { inject, reactive, watch } from "vue";
import type { ApiClient } from "../api/client";
import type { Citation, Evidence } from "../api/types";

const props = defineProps<{ citations: Citation[]; open?: number | null }>();
const api = inject<ApiClient>("api")!;

interface Entry {
  expanded: boolean;
  loading: boolean;
  evidence?: Evidence | null;
  error?: string;
}
const entries = reactive<Record<number, Entry>>({});

async function load(c: Citation) {
  if (!entries[c.number]) entries[c.number] = { expanded: false, loading: false };
  const entry = entries[c.number]; // the reactive proxy, so later changes re-render
  entry.expanded = true;
  if (entry.evidence !== undefined || entry.loading) return;
  entry.loading = true;
  try {
    entry.evidence = await api.evidence(c.review_id);
  } catch (error) {
    entry.error = String(error);
  } finally {
    entry.loading = false;
  }
}

function toggle(c: Citation) {
  const entry = entries[c.number];
  if (entry?.expanded) entry.expanded = false;
  else void load(c);
}

function highlighted(c: Citation, evidence: Evidence) {
  const span = evidence.chunks.find((chunk) => chunk.chunk_id === c.chunk_id);
  if (!span) return { before: evidence.text, mark: "", after: "" };
  return { before: evidence.text.slice(0, span.start_offset), mark: evidence.text.slice(span.start_offset, span.end_offset),
           after: evidence.text.slice(span.end_offset) };
}

watch(() => props.open, (number) => {
  const c = props.citations.find((x) => x.number === number);
  if (c) void load(c);
}, { immediate: true });
watch(() => props.citations, () => Object.keys(entries).forEach((k) => delete entries[Number(k)]));
</script>

<template>
  <section v-if="citations.length" class="card">
    <h2>引用</h2>
    <ul class="citations">
      <li v-for="c in citations" :key="c.number" :data-test="`citation-${c.number}`">
        <button class="citation-head" @click="toggle(c)">
          <strong>[{{ c.number }}]</strong> <code>{{ c.chunk_id }}</code>
          <span class="snippet">{{ c.text.length > 140 ? c.text.slice(0, 139) + "…" : c.text }}</span>
        </button>
        <div v-if="entries[c.number]?.expanded" class="evidence">
          <p v-if="entries[c.number].loading" class="muted">加载原评论…</p>
          <p v-else-if="entries[c.number].error" class="muted">加载失败：{{ entries[c.number].error }}</p>
          <p v-else-if="entries[c.number].evidence === null" class="muted">原评论已不在当前数据中。</p>
          <template v-else-if="entries[c.number].evidence">
            <p class="review-text">{{ highlighted(c, entries[c.number].evidence!).before }}<mark>{{
              highlighted(c, entries[c.number].evidence!).mark }}</mark>{{ highlighted(c, entries[c.number].evidence!).after }}</p>
            <p class="muted">
              评论 ID {{ entries[c.number].evidence!.review_id }}（系统生成，不是 Amazon 评论 ID）·
              {{ entries[c.number].evidence!.helpful_yes }}/{{ entries[c.number].evidence!.helpful_total }} 人觉得有用
            </p>
          </template>
        </div>
      </li>
    </ul>
  </section>
</template>
