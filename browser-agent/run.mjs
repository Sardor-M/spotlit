#!/usr/bin/env node
/** Runs live browser tasks with Claude Code or Codex, once with every element and once with spotlit's top k. */
import { spawn } from "node:child_process";
import { appendFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { claudeCommand, codexCommand, readTasks, ROOT, serverArgs, taskPrompt } from "./src/agents.mjs";

const { values: options } = parseArgs({ options: {
  agent: { type: "string", default: "claude" },
  mode: { type: "string", default: "both" },
  tasks: { type: "string", default: new URL("./tasks.json", import.meta.url).pathname },
  only: { type: "string" },
  model: { type: "string" },
  effort: { type: "string" },
  k: { type: "string", default: "50" },
  repeat: { type: "string", default: "1" },
  "max-turns": { type: "string", default: "40" },
  "timeout-min": { type: "string", default: "15" },
  parallel: { type: "boolean", default: false },
  headless: { type: "boolean", default: false },
  out: { type: "string" },
} });

const pick = (value, all) => (value === "both" ? all : value.split(","));
const agents = pick(options.agent, ["claude", "codex"]);
const modes = pick(options.mode, ["full", "spotlit"]);
const only = options.only?.split(",");
const tasks = readTasks(options.tasks).filter((task) => !only || only.includes(task.id));
const out = options.out ? resolve(options.out) : join(ROOT, "runs", "agent", new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19));
mkdirSync(out, { recursive: true });

function agentCommand(agent, task, mode, dir, slot, cwd) {
  const side = options.parallel ? ["left", "right"][slot] : "";
  const server = serverArgs({ mode, log: join(dir, "tools.jsonl"), k: options.k, side, headless: options.headless });
  const prompt = taskPrompt(task);
  if (agent === "claude") return { ...claudeCommand({ server, configDir: dir, model: options.model, maxTurns: options["max-turns"], prompt }), output: "claude.json" };
  return { ...codexCommand({ server, cwd, model: options.model, effort: options.effort, prompt }), output: "codex.jsonl" };
}

function runProcess({ command, args, env }, cwd, stdoutPath, stderrPath) {
  return new Promise((resolve) => {
    const child = spawn(command, args, { cwd, env: { ...process.env, ...env }, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "", stderr = "";
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    const timer = setTimeout(() => child.kill("SIGTERM"), Number(options["timeout-min"]) * 60000);
    child.on("close", (code, signal) => {
      clearTimeout(timer);
      writeFileSync(stdoutPath, stdout);
      writeFileSync(stderrPath, stderr);
      resolve({ code, signal, stdout });
    });
  });
}

function parseClaude(stdout) {
  const result = JSON.parse(stdout.trim().split("\n").at(-1));
  const usage = result.usage ?? {};
  return {
    final: result.result ?? "", error: result.is_error ? result.subtype : null, turns: result.num_turns,
    input_tokens: (usage.input_tokens ?? 0) + (usage.cache_creation_input_tokens ?? 0) + (usage.cache_read_input_tokens ?? 0),
    output_tokens: usage.output_tokens ?? 0, cost_usd: result.total_cost_usd ?? null,
  };
}

function parseCodex(stdout) {
  const parsed = { final: "", error: null, turns: 0, input_tokens: 0, output_tokens: 0, cost_usd: null };
  for (const line of stdout.split("\n")) {
    if (!line.startsWith("{")) continue;
    const event = JSON.parse(line);
    if (event.type === "item.completed" && event.item?.type === "agent_message") parsed.final = event.item.text ?? "";
    if (event.type === "item.completed" && event.item?.type === "mcp_tool_call") parsed.turns += 1;
    if (event.type === "turn.completed") {
      parsed.input_tokens += event.usage?.input_tokens ?? 0;
      parsed.output_tokens += event.usage?.output_tokens ?? 0;
    }
    if (event.type === "turn.failed" || event.type === "error") parsed.error = event.error?.message ?? event.message ?? "error";
  }
  return parsed;
}

const readLog = (path) => (existsSync(path) ? readFileSync(path, "utf8").trim().split("\n").filter(Boolean).map((line) => JSON.parse(line)) : []);
const norm = (text) => { try { return decodeURIComponent(text); } catch { return text; } };

async function runOne(task, agent, mode, attempt, slot) {
  const dir = join(out, `${task.id}-${agent}-${mode}${Number(options.repeat) > 1 ? `-${attempt}` : ""}`);
  mkdirSync(dir, { recursive: true });
  const cwd = mkdtempSync(join(tmpdir(), "spotlit-agent-"));
  const command = agentCommand(agent, task, mode, dir, slot, cwd);
  const start = Date.now();
  const { code, signal, stdout } = await runProcess(command, cwd, join(dir, command.output), join(dir, "stderr.txt"));
  rmSync(cwd, { recursive: true, force: true });
  let parsed;
  try {
    parsed = agent === "claude" ? parseClaude(stdout) : parseCodex(stdout);
  } catch {
    parsed = { final: "", error: signal ? `killed (${signal})` : `exit ${code}, unparsable output`, turns: null, input_tokens: null, output_tokens: null, cost_usd: null };
  }
  const calls = readLog(join(dir, "tools.jsonl"));
  const lastUrl = calls.findLast((call) => call.url)?.url ?? "";
  const answer = parsed.final.match(/ANSWER:\s*(.*)/i)?.[1]?.trim() ?? "";
  const checks = [];
  if (task.check.url) checks.push(norm(lastUrl).includes(norm(task.check.url)));
  for (const part of [task.check.answer ?? []].flat()) checks.push(parsed.final.toLowerCase().includes(part.toLowerCase()));
  const elementCalls = calls.filter((call) => call.tool === "elements" && call.ok);
  const row = {
    task: task.id, agent, mode, attempt, success: checks.length > 0 && checks.every(Boolean), answer: answer.slice(0, 120),
    status: null,
    last_url: lastUrl, tool_calls: calls.length, elements_calls: elementCalls.length,
    observation_chars: elementCalls.reduce((sum, call) => sum + call.chars, 0),
    max_elements: Math.max(0, ...elementCalls.map((call) => call.elements_total)),
    wall_s: Math.round((Date.now() - start) / 1000), ...parsed, final: undefined,
  };
  row.status = row.success ? "pass" : row.error ? "error" : "fail";
  appendFileSync(join(out, "results.jsonl"), JSON.stringify(row) + "\n");
  const tokens = row.input_tokens === null ? "?" : row.input_tokens.toLocaleString("en-US");
  console.log(`${row.status.toUpperCase().padEnd(5)} ${task.id}  ${agent}/${mode}  input ${tokens} tokens  ${row.tool_calls} tool calls  ${row.wall_s}s${row.error ? `  (${row.error})` : ""}`);
  return row;
}

const median = (values) => {
  const sorted = values.filter((v) => typeof v === "number").sort((a, b) => a - b);
  return sorted.length ? sorted[Math.floor((sorted.length - 1) / 2)] : null;
};
const fmt = (value) => (value === null || value === undefined ? "-" : typeof value === "number" ? value.toLocaleString("en-US") : String(value));

function summary(rows) {
  const lines = ["| task | agent | mode | result | input tokens | output tokens | tool calls | elements shown (chars) | time (s) | cost (USD) |",
    "| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |"];
  for (const r of rows) {
    lines.push(`| ${r.task} | ${r.agent} | ${r.mode} | ${r.status} | ${fmt(r.input_tokens)} | ${fmt(r.output_tokens)} | ${r.tool_calls} | ${fmt(r.observation_chars)} | ${r.wall_s} | ${r.cost_usd === null ? "-" : r.cost_usd.toFixed(3)} |`);
  }
  lines.push("", "Runs that ended in an agent error (for example a usage limit) are left out of passed and the token columns.", "",
    "| agent | mode | passed | median input tokens | total input tokens | median time (s) | total cost (USD) | errors |",
    "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |");
  for (const agent of agents) for (const mode of modes) {
    const all = rows.filter((r) => r.agent === agent && r.mode === mode);
    const group = all.filter((r) => r.status !== "error");
    if (!all.length) continue;
    const costs = group.map((r) => r.cost_usd).filter((c) => c !== null);
    lines.push(`| ${agent} | ${mode} | ${group.filter((r) => r.success).length}/${group.length} | ${fmt(median(group.map((r) => r.input_tokens)))} | ${fmt(group.reduce((s, r) => s + (r.input_tokens ?? 0), 0))} | ${fmt(median(group.map((r) => r.wall_s)))} | ${costs.length ? costs.reduce((a, b) => a + b, 0).toFixed(2) : "-"} | ${all.length - group.length} |`);
  }
  return lines.join("\n");
}

const rows = [];
for (let attempt = 1; attempt <= Number(options.repeat); attempt++) {
  for (const task of tasks) for (const agent of agents) {
    if (options.parallel) rows.push(...await Promise.all(modes.map((mode, slot) => runOne(task, agent, mode, attempt, slot))));
    else for (const mode of modes) rows.push(await runOne(task, agent, mode, attempt, 0));
  }
}
const table = summary(rows);
writeFileSync(join(out, "summary.md"), table + "\n");
console.log(`\n${table}\n\nResults: ${out}`);
