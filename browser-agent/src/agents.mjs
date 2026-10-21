import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { basename, join } from "node:path";

export const MCP = new URL("../mcp.mjs", import.meta.url).pathname;
export const ROOT = new URL("../../", import.meta.url).pathname;

/** Tasks from a JSON file, looked up from the working directory, else by name in browser-agent/. */
export function readTasks(file) {
  const path = existsSync(file) ? file : new URL(`../${basename(file)}`, import.meta.url).pathname;
  return JSON.parse(readFileSync(path, "utf8"));
}

// Codex's own browser, plugins and shell would let it skip the browser server under test.
export const CODEX_OFF = ["plugins", "apps", "browser_use", "browser_use_external", "in_app_browser", "computer_use",
  "image_generation", "shell_tool", "unified_exec", "multi_agent", "tool_suggest"];

export const taskPrompt = (task) => [
  "You control a Chrome browser through the browser tools: open, elements, click, type, select, press, back.",
  `Task: ${task.task}`,
  `Start by opening ${task.start}`,
  "Use open only for that start page. After it, move around the site only by clicking, typing, selecting and pressing keys on the page.",
  "Use only the browser tools. Do not run shell commands or search the web.",
  "Element ids change whenever the page changes, so call elements again after every action.",
  'When the task is done, end your reply with one line: ANSWER: <the answer, or "done">',
].join("\n");

const WINDOWS = { left: ["--position", "0,0", "--size", "960,1000"], right: ["--position", "960,0", "--size", "960,1000"] };

/** Arguments for `node mcp.mjs`; `side` is left, right, or empty for the default window. */
export function serverArgs({ mode, log, k = "50", side = "", headless = false }) {
  return [MCP, "--mode", mode, "--k", String(k), ...(log ? ["--log", log] : []), ...(WINDOWS[side] ?? []), ...(headless ? ["--headless"] : [])];
}

const waitForFile = (file) => `for i in $(seq 150); do [ -f ${JSON.stringify(file)} ] && break; sleep 0.1; done; sleep 0.5`;

/** Claude Code with only the browser server's tools; `prompt` set means headless (-p) with JSON output. */
export function claudeCommand({ server, configDir, model, maxTurns = "40", prompt }) {
  const config = join(configDir, "mcp.json"), ready = join(configDir, "ready");
  const interactive = prompt === undefined;
  // alwaysLoad keeps the tools out of tool search, which --tools "" removes.
  const browser = { command: process.execPath, args: interactive ? [...server, "--ready", ready] : server, alwaysLoad: true };
  writeFileSync(config, JSON.stringify({ mcpServers: { browser } }, null, 2));
  // --no-chrome: with Claude in Chrome on by default, interactive sessions reach for its tools instead of these.
  const args = ["--mcp-config", config, "--strict-mcp-config", "--no-chrome", "--tools", "", "--allowedTools", "mcp__browser"];
  // Interactive sessions can send the first prompt before MCP tools load; this start-up hook waits for the tool list.
  if (interactive) args.push("--settings", JSON.stringify({ hooks: { SessionStart: [{ hooks: [{ type: "command", command: waitForFile(ready) }] }] } }));
  if (model) args.push("--model", model);
  if (prompt !== undefined) args.unshift("-p", prompt, "--output-format", "json", "--no-session-persistence", "--max-turns", String(maxTurns));
  return { command: process.env.CLAUDE_BIN ?? "claude", args, env: { MAX_MCP_OUTPUT_TOKENS: "200000" } };
}

/** Codex with only the browser server's tools; `exec` means headless with JSONL events. */
export function codexCommand({ server, cwd, model, effort, prompt, exec = true }) {
  const toml = (value) => JSON.stringify(value);
  const args = [
    ...(exec ? ["exec", "--json", "--ephemeral", "--skip-git-repo-check", "-s", "read-only"] : ["-s", "read-only"]), "-C", cwd,
    "-c", "notify=[]", "-c", 'web_search="disabled"', "-c", "tool_output_token_limit=200000",
    "-c", `mcp_servers.browser.command=${toml(process.execPath)}`, "-c", `mcp_servers.browser.args=${toml(server)}`,
    "-c", "mcp_servers.browser.startup_timeout_sec=60", "-c", "mcp_servers.browser.tool_timeout_sec=180",
    "-c", 'mcp_servers.browser.default_tools_approval_mode="approve"',
    ...CODEX_OFF.flatMap((feature) => ["--disable", feature]),
  ];
  if (model) args.push("-m", model);
  if (effort) args.push("-c", `model_reasoning_effort=${toml(effort)}`);
  if (prompt) args.push(prompt);
  return { command: process.env.CODEX_BIN ?? "codex", args, env: {} };
}
