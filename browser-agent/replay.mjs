#!/usr/bin/env node
/** Mind2Web dev replay: the same LLM picks the next element from every element, or from spotlit's top k. */
import { spawn } from "node:child_process";
import { appendFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { CODEX_OFF, ROOT } from "./src/agents.mjs";
import { loadRanker } from "./src/ranker.mjs";

const { values: options } = parseArgs({ options: {
  agent: { type: "string", default: "claude" },
  arms: { type: "string", default: "full,spotlit" },
  n: { type: "string", default: "20" },
  seed: { type: "string", default: "1" },
  k: { type: "string", default: "50" },
  model: { type: "string" },
  effort: { type: "string" },
  concurrency: { type: "string", default: "3" },
  out: { type: "string" },
} });

const DEV = join(ROOT, "data", "mind2web", "dev.jsonl");
const K = Number(options.k), arms = options.arms.split(",");
const out = options.out ? resolve(options.out) : join(ROOT, "runs", "agent", `replay-${options.agent}-${new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)}`);
mkdirSync(out, { recursive: true });

function sample(rows, n, seed) {
  let state = seed >>> 0;
  const random = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const order = rows.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  return order.slice(0, n).map((i) => rows[i]);
}

const SYSTEM = "You choose the next action of a web agent on a web page. Reply with one JSON object and nothing else.";
const prompt = (step, header, listed) => [
  `Task: ${step.task}`,
  `Previous actions: ${step.history.length ? step.history.join("; ") : "none"}`,
  header,
  ...listed.map(({ id, text }) => `[${id}] ${text}`),
  "",
  'Reply with JSON only: {"id": "<element id>", "action": "CLICK" | "TYPE" | "SELECT", "value": "<text to type or option to select; empty for CLICK>"}',
].join("\n");

function command(cwd) {
  if (options.agent === "claude") {
    const args = ["-p", "--output-format", "json", "--tools", "", "--strict-mcp-config", "--no-session-persistence", "--system-prompt", SYSTEM];
    if (options.model) args.push("--model", options.model);
    return { command: process.env.CLAUDE_BIN ?? "claude", args };
  }
  const args = ["exec", "--json", "--ephemeral", "--skip-git-repo-check", "-s", "read-only", "-C", cwd, "-c", "notify=[]",
    "-c", 'web_search="disabled"', ...CODEX_OFF.flatMap((feature) => ["--disable", feature])];
  if (options.model) args.push("-m", options.model);
  if (options.effort) args.push("-c", `model_reasoning_effort=${JSON.stringify(options.effort)}`);
  return { command: process.env.CODEX_BIN ?? "codex", args: [...args, "-"] };
}

function ask(text) {
  const cwd = mkdtempSync(join(tmpdir(), "spotlit-replay-"));
  const { command: bin, args } = command(cwd);
  const input = options.agent === "claude" ? text : `${SYSTEM}\n\n${text}`;
  return new Promise((done) => {
    const child = spawn(bin, args, { cwd, stdio: ["pipe", "pipe", "pipe"] });
    let stdout = "", stderr = "";
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("close", (code) => { rmSync(cwd, { recursive: true, force: true }); done({ code, stdout, stderr }); });
    child.stdin.end(input);
  });
}

function parseReply({ code, stdout, stderr }) {
  let reply = "", input_tokens = null, output_tokens = null, cost_usd = null, error = null;
  try {
    if (options.agent === "claude") {
      const result = JSON.parse(stdout.trim().split("\n").at(-1));
      const usage = result.usage ?? {};
      reply = result.result ?? "";
      input_tokens = (usage.input_tokens ?? 0) + (usage.cache_creation_input_tokens ?? 0) + (usage.cache_read_input_tokens ?? 0);
      output_tokens = usage.output_tokens ?? 0;
      cost_usd = result.total_cost_usd ?? null;
      if (result.is_error) error = reply.slice(0, 200) || result.subtype;
    } else {
      input_tokens = 0; output_tokens = 0;
      for (const line of stdout.split("\n").filter((l) => l.startsWith("{"))) {
        const event = JSON.parse(line);
        if (event.type === "item.completed" && event.item?.type === "agent_message") reply = event.item.text ?? "";
        if (event.type === "turn.completed") { input_tokens += event.usage?.input_tokens ?? 0; output_tokens += event.usage?.output_tokens ?? 0; }
        if (event.type === "turn.failed" || event.type === "error") error = event.error?.message ?? event.message ?? "error";
      }
    }
  } catch {
    error = `exit ${code}: ${stderr.slice(-200)}`;
  }
  let picked = {};
  try { picked = JSON.parse(reply.match(/\{[\s\S]*\}/)?.[0] ?? "{}"); } catch { error ??= `unparsable reply: ${reply.slice(0, 120)}`; }
  return { picked, input_tokens, output_tokens, cost_usd, error };
}

const clean = (value) => String(value ?? "").trim().toLowerCase();

async function runStep(step, arm, ranker) {
  const candidates = step.candidates.map(([id, text]) => ({ id, text }));
  let header, listed;
  if (arm === "full") {
    listed = [...candidates].sort((a, b) => Number(a.id) - Number(b.id));
    header = `The page's ${listed.length} elements, in page order, one per line as [id] description:`;
  } else {
    listed = await ranker.rank(step.task, candidates, { k: K, history: step.history });
    header = `The ${listed.length} elements most relevant to the task, out of the page's ${candidates.length}, best first, one per line as [id] description:`;
  }
  const text = prompt(step, header, listed);
  const start = Date.now();
  const reply = parseReply(await ask(text));
  const gold = new Set(step.gold);
  const id = String(reply.picked.id ?? "").replace(/[[\]]/g, "").trim();
  const action = clean(reply.picked.action).toUpperCase();
  const elementOk = gold.has(id);
  const opOk = action === step.op && (step.op === "CLICK" || clean(reply.picked.value) === clean(step.value));
  return {
    action_uid: step.action_uid, website: step.website, arm, op: step.op, gold: step.gold, listed: listed.length,
    gold_listed: listed.some((c) => gold.has(c.id)), prompt_chars: text.length, picked: id, picked_action: action,
    element_ok: elementOk, step_ok: elementOk && opOk, seconds: Math.round((Date.now() - start) / 100) / 10,
    input_tokens: reply.input_tokens, output_tokens: reply.output_tokens, cost_usd: reply.cost_usd, error: reply.error,
  };
}

async function pool(jobs, size) {
  const results = [];
  let next = 0;
  await Promise.all(Array.from({ length: size }, async () => {
    while (next < jobs.length) {
      const job = jobs[next++];
      results.push(await job());
    }
  }));
  return results;
}

const steps = sample(readFileSync(DEV, "utf8").trim().split("\n").map((line) => JSON.parse(line)), Number(options.n), Number(options.seed));
const ranker = arms.includes("spotlit") ? await loadRanker() : null;
const jobs = steps.flatMap((step) => arms.map((arm) => async () => {
  const row = await runStep(step, arm, ranker);
  appendFileSync(join(out, "results.jsonl"), JSON.stringify(row) + "\n");
  console.log(`${row.element_ok ? "right" : "wrong"}  ${arm.padEnd(7)}  ${row.website.padEnd(16)}  ${String(row.listed).padStart(5)} listed  ${row.input_tokens ?? "?"} tokens  ${row.seconds}s${row.error ? `  (${row.error})` : ""}`);
  return row;
}));
const rows = await pool(jobs, Number(options.concurrency));

const median = (values) => { const s = values.filter((v) => typeof v === "number").sort((a, b) => a - b); return s.length ? s[Math.floor((s.length - 1) / 2)] : null; };
const pct = (part, whole) => (whole ? `${part}/${whole} (${Math.round((100 * part) / whole)}%)` : "-");
const fmt = (value) => (value === null ? "-" : value.toLocaleString("en-US"));
const ok = rows.filter((r) => !r.error);
const lines = [`Mind2Web dev, ${steps.length} steps (seed ${options.seed}), agent ${options.agent}${options.model ? ` (${options.model})` : ""}, spotlit k = ${K}. Steps that errored are left out of each arm's rates.`, "",
  "| arm | right element | right element and action | gold in the list | median input tokens | total input tokens | median seconds | cost (USD) | errors |",
  "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"];
for (const arm of arms) {
  const group = ok.filter((r) => r.arm === arm);
  const costs = group.map((r) => r.cost_usd).filter((c) => c !== null);
  lines.push(`| ${arm} | ${pct(group.filter((r) => r.element_ok).length, group.length)} | ${pct(group.filter((r) => r.step_ok).length, group.length)} | ${pct(group.filter((r) => r.gold_listed).length, group.length)} | ${fmt(median(group.map((r) => r.input_tokens)))} | ${fmt(group.reduce((s, r) => s + (r.input_tokens ?? 0), 0))} | ${median(group.map((r) => r.seconds))} | ${costs.length ? costs.reduce((a, b) => a + b, 0).toFixed(2) : "-"} | ${rows.filter((r) => r.arm === arm && r.error).length} |`);
}
if (arms.includes("full") && arms.includes("spotlit")) {
  const byStep = Map.groupBy(ok, (r) => r.action_uid);
  const pairs = [...byStep.values()].filter((pair) => pair.length === 2).map((pair) => Object.fromEntries(pair.map((r) => [r.arm, r])));
  const count = (test) => pairs.filter(test).length;
  const missed = pairs.filter((p) => !p.spotlit.gold_listed);
  lines.push("", `Paired over ${pairs.length} steps with no error in either arm: both right ${count((p) => p.full.element_ok && p.spotlit.element_ok)}, only full right ${count((p) => p.full.element_ok && !p.spotlit.element_ok)}, only spotlit right ${count((p) => !p.full.element_ok && p.spotlit.element_ok)}, both wrong ${count((p) => !p.full.element_ok && !p.spotlit.element_ok)}. The gold element was outside spotlit's top ${K} on ${missed.length} of them; the full list got ${missed.filter((p) => p.full.element_ok).length} of those right.`);
}
writeFileSync(join(out, "summary.md"), lines.join("\n") + "\n");
console.log(`\n${lines.join("\n")}\n\nResults: ${out}`);
