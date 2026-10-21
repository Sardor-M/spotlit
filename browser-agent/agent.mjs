#!/usr/bin/env node
/** Opens an interactive Claude Code or Codex session wired to the browser server in one mode, then sums its log. */
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { claudeCommand, codexCommand, readTasks, ROOT, serverArgs, taskPrompt } from "./src/agents.mjs";

const TASK_FILES = ["./tasks.json", "./demo-tasks.json"].map((file) => new URL(file, import.meta.url).pathname);
const { values: options, positionals: [agent, mode] } = parseArgs({ allowPositionals: true, options: {
  side: { type: "string", default: "" },
  task: { type: "string" },
  tasks: { type: "string" },
  model: { type: "string" },
  effort: { type: "string" },
  k: { type: "string", default: "50" },
} });
if (!["claude", "codex"].includes(agent) || !["full", "spotlit"].includes(mode)) {
  console.error("usage: node browser-agent/agent.mjs <claude|codex> <full|spotlit> [--side left|right] [--task id] [--model m]");
  process.exit(2);
}

const workspace = join(ROOT, "runs", "agent", "workspace");
const dir = join(ROOT, "runs", "agent", "manual", `${new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)}-${agent}-${mode}`);
mkdirSync(workspace, { recursive: true });
mkdirSync(dir, { recursive: true });
const log = join(dir, "tools.jsonl");
const server = serverArgs({ mode, log, k: options.k, side: options.side });
let prompt;
if (options.task) {
  const tasks = (options.tasks ? [options.tasks] : TASK_FILES).flatMap(readTasks);
  const task = tasks.find((t) => t.id === options.task);
  if (!task) throw new Error(`no task ${options.task}; known ids: ${tasks.map((t) => t.id).join(", ")}`);
  prompt = taskPrompt(task);
}
const command = agent === "claude"
  ? claudeCommand({ server, configDir: dir, model: options.model })
  : codexCommand({ server, cwd: workspace, model: options.model, effort: options.effort, exec: false });
if (prompt) command.args.push(prompt);

const child = spawn(command.command, command.args, { cwd: workspace, env: { ...process.env, ...command.env }, stdio: "inherit" });
child.on("close", (code) => {
  const calls = existsSync(log) ? readFileSync(log, "utf8").trim().split("\n").filter(Boolean).map((line) => JSON.parse(line)) : [];
  const listings = calls.filter((call) => call.tool === "elements" && call.ok);
  const chars = listings.reduce((sum, call) => sum + call.chars, 0);
  console.log([
    `\n${agent} / ${mode}: ${calls.length} browser tool calls, ${listings.length} element listings`,
    `page listings sent to the model: ${chars.toLocaleString("en-US")} characters (about ${Math.round(chars / 4).toLocaleString("en-US")} tokens)`,
    `last page: ${calls.findLast((call) => call.url)?.url ?? "-"}`,
    `log: ${log}`,
  ].join("\n"));
  process.exit(code ?? 0);
});
