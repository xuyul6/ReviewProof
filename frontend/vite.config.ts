import vue from "@vitejs/plugin-vue";
import { defineConfig } from "vitest/config";

const api = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [vue()],
  server: { proxy: { "/v1": api, "/healthz": api } },
  test: { environment: "jsdom" },
});
