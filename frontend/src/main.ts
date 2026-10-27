import { createApp } from "vue";
import "./style.css";
import App from "./App.vue";
import { createApiClient } from "./api/client";

createApp(App).provide("api", createApiClient()).mount("#app");
