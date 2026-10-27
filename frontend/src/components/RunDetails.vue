<script setup lang="ts">
import { inject, ref } from "vue";
import type { ApiClient } from "../api/client";
import type { DoneEvent, RetrievalSnapshot, TraceEvent } from "../api/types";

const props = defineProps<{ done: DoneEvent; progress: TraceEvent[] }>();
const api = inject<ApiClient>("api")!;

const retrievals = ref<RetrievalSnapshot[] | null>(null);
const loadError = ref(false);
let requested = false;

async function onToggle(event: Event) {
  if (!(event.target as HTMLDetailsElement).open || requested) return;
  requested = true;
  try {
    retrievals.value = (await api.run(props.done.run_id)).retrievals ?? [];
  } catch {
    loadError.value = true;
  }
}

const fmt = (value: number | null, digits: number) => (value === null ? "—" : value.toFixed(digits));
</script>

<template>
  <details class="card" data-test="run-details" @toggle="onToggle">
    <summary>运行详情</summary>
    <dl>
      <dt>run_id</dt><dd><code>{{ done.run_id }}</code>{{ done.persisted ? "" : "（未能保存）" }}</dd>
      <dt>检索词</dt><dd>{{ done.search_queries.join(" → ") || "—" }}</dd>
      <dt>检索模式</dt><dd>{{ done.retrieval_modes.join(" → ") || "—" }}</dd>
      <dt>改写次数</dt><dd>{{ done.rewrites_used }}</dd>
      <dt>LLM 调用</dt><dd>{{ done.llm_calls }} 次，输入 {{ done.input_tokens }} / 输出 {{ done.output_tokens }} token</dd>
      <dt>总耗时</dt><dd>{{ (done.latency_ms / 1000).toFixed(1) }} s</dd>
    </dl>
    <table>
      <thead><tr><th>节点</th><th>状态</th><th>耗时</th><th>说明</th></tr></thead>
      <tbody>
        <tr v-for="(t, i) in progress" :key="i">
          <td>{{ t.node }}</td><td>{{ t.status }}</td><td>{{ t.latency_ms.toFixed(0) }} ms</td><td>{{ t.detail }}</td>
        </tr>
      </tbody>
    </table>
    <p v-if="loadError" class="muted">运行记录暂时取不到。</p>
    <template v-for="(snap, i) in retrievals ?? []" :key="i">
      <h3 class="snapshot-title">检索 {{ i + 1 }}：{{ snap.query }} · {{ snap.strategy }} · {{ snap.mode }}</h3>
      <table v-if="snap.candidates.length">
        <thead><tr><th>#</th><th>chunk_id</th><th>向量</th><th>BM25</th><th>RRF</th><th>重排</th></tr></thead>
        <tbody>
          <tr v-for="(c, j) in snap.candidates" :key="c.chunk_id">
            <td>{{ j + 1 }}</td><td><code>{{ c.chunk_id }}</code></td>
            <td>{{ c.dense_rank ?? "—" }}</td><td>{{ c.bm25_rank ?? "—" }}</td>
            <td>{{ fmt(c.rrf_score, 4) }}</td><td>{{ fmt(c.rerank_score, 3) }}</td>
          </tr>
        </tbody>
      </table>
    </template>
  </details>
</template>
