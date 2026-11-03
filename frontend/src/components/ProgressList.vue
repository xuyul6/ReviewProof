<script setup lang="ts">
import type { TraceEvent } from "../api/types";

defineProps<{ progress: TraceEvent[]; running: boolean }>();

const NODE_LABELS: Record<string, string> = {
  plan: "规划", retrieve: "检索", assess: "评估证据", rewrite: "改写重检索", generate: "生成回答", verify: "核验引用",
  advise: "补充参考",
};
const STATUS_LABELS: Record<string, string> = { ok: "完成", degraded: "降级", failed: "失败", skipped: "跳过" };
</script>

<template>
  <ol class="progress" data-test="progress">
    <li v-for="(step, i) in progress" :key="i" :class="`step-${step.status}`">
      <span class="node">{{ NODE_LABELS[step.node] ?? step.node }}</span>
      <span class="status">{{ STATUS_LABELS[step.status] ?? step.status }}</span>
      <span class="ms">{{ (step.latency_ms / 1000).toFixed(1) }} s</span>
    </li>
    <li v-if="running" class="step-running"><span class="spinner" aria-hidden="true"></span>处理中…</li>
  </ol>
</template>
