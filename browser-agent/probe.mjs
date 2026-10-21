#!/usr/bin/env node
/** No-LLM check of each task's start page: element count, full-list size, and spotlit's rank for the first target (ranked by `goal`, else the task). */
import { parseArgs } from "node:util";
import { readTasks } from "./src/agents.mjs";
import { Browser } from "./src/browser.mjs";
import { loadRanker } from "./src/ranker.mjs";

const { values: options } = parseArgs({ options: {
  tasks: { type: "string", default: new URL("./tasks.json", import.meta.url).pathname },
  only: { type: "string" }, headless: { type: "boolean", default: false },
} });
const only = options.only?.split(",");
const tasks = readTasks(options.tasks).filter((task) => !only || only.includes(task.id));
const ranker = await loadRanker();
const browser = await Browser.start({ headless: options.headless });
const rows = [];
try {
  for (const task of tasks) {
    const where = await browser.open(task.start);
    const snap = await browser.snapshot();
    const listing = snap.candidates.map(([id, text]) => `[${id}] ${text}`).join("\n");
    const start = performance.now();
    const ranked = await ranker.rank(task.goal ?? task.task, snap.candidates.map(([id, text]) => ({ id, text })), { k: snap.candidates.length });
    const rankMs = Math.round(performance.now() - start);
    const needle = task.probe.toLowerCase();
    const hit = ranked.findIndex((c) => c.text.toLowerCase().includes(needle));
    rows.push({ task: task.id, title: where.title.slice(0, 40), elements: snap.candidates.length,
      full_chars: listing.length, top50_chars: ranked.slice(0, 50).map((c) => `[${c.id}] ${c.text}`).join("\n").length,
      matches: ranked.filter((c) => c.text.toLowerCase().includes(needle)).length,
      best_match_rank: hit < 0 ? null : hit + 1, best_match: hit < 0 ? "" : ranked[hit].text.slice(0, 70), rank_ms: rankMs });
  }
} finally {
  browser.close();
}
console.table(rows);
