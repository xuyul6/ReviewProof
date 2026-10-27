import { createApp } from "vue";
import App from "./App.vue";
import { createApiClient } from "./api/client";

createApp(App).provide("api", createApiClient()).mount("#app");
