import { createApp } from "vue";
import "./style.css";
import App from "./App.vue";
import { createAnnotationClient } from "./api/annotation";
import { createApiClient } from "./api/client";
import AnnotateApp from "./annotate/AnnotateApp.vue";

// #/annotate?set=gold_v1 opens the local annotation review page (API needs REVIEWPROOF_ANNOTATION=1).
const hash = window.location.hash;
if (hash.startsWith("#/annotate")) {
  const labelSet = new URLSearchParams(hash.split("?")[1] ?? "").get("set") || "gold_v1";
  createApp(AnnotateApp, { labelSet }).provide("annotationApi", createAnnotationClient()).mount("#app");
} else {
  createApp(App).provide("api", createApiClient()).mount("#app");
}
