<script setup lang="ts">
import type { DoneEvent, TraceEvent } from "../api/types";

defineProps<{ done: DoneEvent; progress: TraceEvent[] }>();
</script>

<template>
  <details class="card" data-test="run-details">
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
  </details>
</template>
