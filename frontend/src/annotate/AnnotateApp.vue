<script setup lang="ts">
import { computed, inject, onBeforeUnmount, onMounted, ref } from "vue";
import type {
  AnnotationClient, AnnotationItem, Answerability, Grade, ItemSummary, PoolChunk, Verdict,
} from "../api/annotation";
import { ApiError } from "../api/client";
import ChunkGrader from "./ChunkGrader.vue";

const props = defineProps<{ labelSet: string }>();
const api = inject<AnnotationClient>("annotationApi")!;

const ANSWERABILITY: [Answerability, string][] = [["answered", "能回答"], ["partial", "部分"], ["unanswerable", "不能"]];
const VERDICTS: Verdict[] = ["yes", "no", "mixed", "unknown"];

const items = ref<ItemSummary[]>([]);
const splitFilter = ref("");
const statusFilter = ref("");
const current = ref<AnnotationItem | null>(null);
const answerability = ref<Answerability | null>(null);
const verdict = ref<Verdict | null>(null);
const grades = ref<Record<string, Grade>>({});
const rationale = ref("");
const prominent = ref<Set<string>>(new Set());
const extras = ref<PoolChunk[]>([]);
const searchQuery = ref("");
const searchHits = ref<PoolChunk[]>([]);
const error = ref("");
const busy = ref(false);
// Ctrl+Enter is ignored this long after an item opens, so a double tap cannot accept a draft unseen.
const SAVE_GUARD_MS = 500;
let openedAt = 0;

const blindMode = computed(() => current.value?.draft_hidden ?? false);
const isYesNo = computed(() => current.value?.question_type === "yesno");
const shown = computed(() => items.value.filter((i) =>
  (!splitFilter.value || i.split === splitFilter.value) && (!statusFilter.value || i.status === statusFilter.value)));
const doneCount = computed(() => items.value.filter((i) => i.status === "done").length);
const topChunks = computed(() => current.value?.pool.filter((c) => prominent.value.has(c.chunk_id)) ?? []);
const restChunks = computed(() => current.value?.pool.filter((c) => !prominent.value.has(c.chunk_id)) ?? []);

function message(e: unknown): string {
  return e instanceof ApiError ? e.detail : String(e);
}

function fill(item: AnnotationItem) {
  current.value = item;
  openedAt = performance.now();
  error.value = "";
  searchHits.value = [];
  searchQuery.value = "";
  const source = item.labels.final ?? item.labels.draft;
  if (item.draft_hidden || !source) {
    answerability.value = null;
    verdict.value = null;
    grades.value = {};
    rationale.value = "";
    prominent.value = new Set();
    extras.value = [];
    if (item.draft_hidden) return;
  }
  const base = source?.grades ?? {};
  grades.value = Object.fromEntries([
    ...item.pool.map((c) => [c.chunk_id, base[c.chunk_id] ?? 0]),
    ...item.extra_chunks.filter((c) => c.chunk_id in base).map((c) => [c.chunk_id, base[c.chunk_id]]),
  ]) as Record<string, Grade>;
  extras.value = item.extra_chunks.filter((c) => c.chunk_id in base);
  // Order is fixed when the item opens, so chunks do not jump around while grading. Without a
  // draft nothing was judged irrelevant yet, so every chunk stays unfolded.
  prominent.value = new Set(item.pool.filter((c) => !source || (base[c.chunk_id] ?? 0) >= 1).map((c) => c.chunk_id));
  if (source) {
    answerability.value = source.answerability;
    verdict.value = source.verdict;
    rationale.value = source.rationale;
  }
}

async function open(qid: string) {
  try {
    fill(await api.item(props.labelSet, qid));
  } catch (e) {
    error.value = message(e);
  }
}

function nextPendingAfter(qid: string | undefined): string | undefined {
  const list = items.value;
  const start = list.findIndex((i) => i.qid === qid);
  for (let step = 1; step <= list.length; step++) {
    const candidate = list[(start + step + list.length) % list.length];
    if (candidate.status !== "done" && candidate.qid !== qid) return candidate.qid;
  }
  return undefined;
}

async function refreshList() {
  items.value = await api.items(props.labelSet);
}

function choose(value: Answerability) {
  answerability.value = value;
  if (isYesNo.value && value === "unanswerable") verdict.value = "unknown";
  else if (verdict.value === "unknown") verdict.value = null;
}

async function submitBlind() {
  if (busy.value || !current.value || !answerability.value) return;
  busy.value = true;
  try {
    fill(await api.blind(props.labelSet, current.value.qid, answerability.value, isYesNo.value ? verdict.value : null));
    await refreshList();
  } catch (e) {
    error.value = message(e);
  } finally {
    busy.value = false;
  }
}

async function saveAndNext() {
  if (busy.value || !current.value || !answerability.value || blindMode.value) return;
  busy.value = true;
  const qid = current.value.qid;
  try {
    await api.final(props.labelSet, qid, {
      answerability: answerability.value, verdict: isYesNo.value ? verdict.value : null,
      grades: { ...grades.value }, rationale: rationale.value,
    });
    await refreshList();
    const next = nextPendingAfter(qid);
    if (next) await open(next);
    else fill(await api.item(props.labelSet, qid));
  } catch (e) {
    error.value = message(e);
  } finally {
    busy.value = false;
  }
}

async function runSearch() {
  if (!current.value || !searchQuery.value.trim()) return;
  try {
    searchHits.value = await api.search(props.labelSet, current.value.qid, searchQuery.value.trim());
  } catch (e) {
    error.value = message(e);
  }
}

function addExtra(chunk: PoolChunk) {
  if (!(chunk.chunk_id in grades.value)) {
    extras.value = [...extras.value, chunk];
    grades.value = { ...grades.value, [chunk.chunk_id]: 0 };
  }
  searchHits.value = searchHits.value.filter((c) => c.chunk_id !== chunk.chunk_id);
}

function setGrade(chunkId: string, g: Grade) {
  grades.value = { ...grades.value, [chunkId]: g };
}

function onKey(event: KeyboardEvent) {
  const target = event.target as HTMLElement | null;
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    if (event.repeat || busy.value || performance.now() - openedAt < SAVE_GUARD_MS) return;
    if (blindMode.value) void submitBlind();
    else void saveAndNext();
    return;
  }
  if (event.ctrlKey || event.metaKey || event.altKey || event.isComposing) return;
  if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
  const index = ["1", "2", "3"].indexOf(event.key);
  if (index >= 0 && current.value) choose(ANSWERABILITY[index][0]);
}

onMounted(async () => {
  window.addEventListener("keydown", onKey);
  try {
    await refreshList();
    const first = items.value.find((i) => i.status !== "done") ?? items.value[0];
    if (first) await open(first.qid);
  } catch (e) {
    error.value = message(e);
  }
});
onBeforeUnmount(() => window.removeEventListener("keydown", onKey));
</script>

<template>
  <div class="annotate">
    <aside class="annotate-list card">
      <h2>{{ labelSet }} · {{ doneCount }}/{{ items.length }}</h2>
      <div class="row">
        <select v-model="splitFilter"><option value="">全部 split</option><option>dev</option><option>test</option></select>
        <select v-model="statusFilter">
          <option value="">全部状态</option><option value="pending">pending</option>
          <option value="drafted">drafted</option><option value="done">done</option>
        </select>
      </div>
      <button type="button" class="next" @click="() => { const n = nextPendingAfter(current?.qid); if (n) open(n); }">
        下一道待审
      </button>
      <ul>
        <li v-for="i in shown" :key="i.qid">
          <button type="button" :class="{ active: current?.qid === i.qid }" @click="open(i.qid)">
            {{ i.qid }} <span class="muted">{{ i.split }} · {{ i.status }}{{ i.blind ? " · 盲标" : "" }}</span>
          </button>
        </li>
      </ul>
    </aside>

    <main v-if="current" class="annotate-main">
      <section class="card">
        <p class="muted">{{ current.qid }} · {{ current.category }} · {{ current.question_type }} · {{ current.split }}
          <span v-if="current.blind"> · 盲标题</span></p>
        <h2>{{ current.question_text }}</h2>
        <div class="row">
          <button v-for="[value, text] in ANSWERABILITY" :key="value" type="button"
                  class="choice" :class="{ active: answerability === value }" :data-test="`ans-${value}`" @click="choose(value)">
            {{ text }}
          </button>
        </div>
        <div v-if="isYesNo" class="row">
          <button v-for="v in VERDICTS" :key="v" type="button" class="choice" :class="{ active: verdict === v }"
                  :data-test="`verdict-${v}`" @click="verdict = v">{{ v }}</button>
        </div>
        <p v-if="blindMode" class="muted">盲标：先独立判断可答性，提交后才显示预标。</p>
      </section>

      <section class="card">
        <template v-if="blindMode">
          <ChunkGrader v-for="c in current.pool" :key="c.chunk_id" :chunk="c" />
        </template>
        <template v-else>
          <ChunkGrader v-for="c in topChunks" :key="c.chunk_id" :chunk="c" :grade="grades[c.chunk_id]"
                       @grade="(g) => setGrade(c.chunk_id, g)" />
          <ChunkGrader v-for="c in extras" :key="c.chunk_id" :chunk="c" :grade="grades[c.chunk_id]" outside
                       @grade="(g) => setGrade(c.chunk_id, g)" />
          <details v-if="restChunks.length">
            <summary>其余 {{ restChunks.length }} 块（预标为无关）</summary>
            <ChunkGrader v-for="c in restChunks" :key="c.chunk_id" :chunk="c" :grade="grades[c.chunk_id]"
                         @grade="(g) => setGrade(c.chunk_id, g)" />
          </details>
          <div class="row search">
            <input v-model="searchQuery" class="grow" placeholder="在这件商品的全部评论里搜索" data-test="search-input"
                   @keydown.enter.exact="runSearch" />
            <button type="button" data-test="search-go" @click="runSearch">搜索</button>
          </div>
          <div v-for="c in searchHits" :key="c.chunk_id" class="search-hit">
            <ChunkGrader :chunk="c" outside />
            <button type="button" :data-test="`add-${c.chunk_id}`" @click="addExtra(c)">加入</button>
          </div>
        </template>
      </section>

      <section class="card">
        <template v-if="!blindMode">
          <label class="muted" for="rationale">理由</label>
          <textarea id="rationale" v-model="rationale" rows="3"></textarea>
        </template>
        <details>
          <summary>社区回答（弱参考，判断以评论为准）</summary>
          <ul><li v-for="(a, n) in current.community_answers" :key="n">{{ a }}</li></ul>
        </details>
        <div class="row">
          <button v-if="blindMode" type="button" data-test="blind-submit" :disabled="busy || !answerability"
                  @click="submitBlind">提交盲标</button>
          <button v-else type="button" data-test="save" :disabled="busy || !answerability" @click="saveAndNext">
            保存并下一题（Ctrl+Enter）
          </button>
          <span v-if="error" class="error" data-test="error">{{ error }}</span>
        </div>
      </section>
    </main>
    <p v-else-if="error" class="error" data-test="error">{{ error }}</p>
  </div>
</template>
