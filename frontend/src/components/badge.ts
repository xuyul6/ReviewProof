import type { DoneEvent } from "../api/types";

export type BadgeTone = "ok" | "partial" | "none" | "error" | "scope";

// Order matters: a service failure is never reported as "the reviews can't confirm this".
export function statusBadge(done: DoneEvent | null): { label: string; tone: BadgeTone } | null {
  if (!done) return null;
  if (done.service_status === "failed") return { label: "服务出错（不代表评论里没有答案）", tone: "error" };
  if (done.route === "out_of_scope") return { label: "不在服务范围", tone: "scope" };
  if (done.status === "answered") return { label: "能回答", tone: "ok" };
  if (done.status === "partial") return { label: "部分能回答", tone: "partial" };
  return { label: "评论无法确认", tone: "none" };
}
