<script setup lang="ts">
import type { Health } from "../api/types";

defineProps<{ health: Health | null; unreachable: boolean }>();
</script>

<template>
  <div v-if="unreachable" class="banner banner-error" data-test="health">
    连不上接口服务。请先在项目目录运行 <code>python -m reviewproof.api</code>，再刷新页面。
  </div>
  <div v-else-if="health && (!health.database || !health.anthropic_key_set)" class="banner banner-warn" data-test="health">
    <span v-if="!health.database">接口连不上数据库，请确认已运行 <code>docker compose up -d db</code>。</span>
    <span v-if="!health.anthropic_key_set">未配置 Claude API key（<code>ANTHROPIC_API_KEY</code>），提问会报服务出错。</span>
  </div>
</template>
