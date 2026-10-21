#!/usr/bin/env node
/** MCP browser server for agents: `--mode full` lists every element, `--mode spotlit` the top k for a goal. */
import { appendFileSync, writeFileSync } from "node:fs";
import { createInterface } from "node:readline";
import { parseArgs } from "node:util";
import { Browser, KEY_NAMES, StaleElementError } from "./src/browser.mjs";

const { values: options } = parseArgs({
  options: {
    mode: { type: "string", default: process.env.SPOTLIT_MODE ?? "spotlit" },
    k: { type: "string", default: process.env.SPOTLIT_K ?? "50" },
    limit: { type: "string", default: "2000" },
    log: { type: "string", default: process.env.SPOTLIT_LOG },
    headless: { type: "boolean", default: process.env.SPOTLIT_HEADLESS === "1" },
    position: { type: "string", default: "0,0" },
    size: { type: "string", default: "1280,900" },
    ready: { type: "string" },
  },
});
const MODE = options.mode;
if (!["full", "spotlit"].includes(MODE)) throw new Error(`--mode must be full or spotlit, not ${MODE}`);
const K = Number(options.k), LIMIT = Number(options.limit), MAX_K = 400;

// Claude Code swaps tool results over about 50K characters for a file path and a 2K preview unless the tool raises this.
const INLINE = { "anthropic/maxResultSizeChars": 600000 };
const idProperty = { type: "string", description: "Element id from the latest elements list, e.g. \"12\"." };
const TOOLS = [
  { name: "open", description: "Open a URL in the browser tab.",
    inputSchema: { type: "object", properties: { url: { type: "string" } }, required: ["url"] } },
  MODE === "spotlit"
    ? { name: "elements",
        description: `List the elements on the current page most relevant to a goal, best first, one per line as [id] description. A small ranking model picks them from every element on the page. Call it again after every action. If what you need is missing, call it with a narrower goal (e.g. "accept cookies button") or a larger k.`,
        inputSchema: { type: "object", properties: {
          goal: { type: "string", description: "What you want to do next on this page, or the whole task." },
          k: { type: "integer", description: `How many elements to return: ${K} unless you ask for more, up to ${MAX_K}.` },
        }, required: ["goal"] }, _meta: INLINE }
    : { name: "elements",
        description: "List every element on the current page in page order, one per line as [id] description. Call it again after every action.",
        inputSchema: { type: "object", properties: {} }, _meta: INLINE },
  { name: "click", description: "Click an element.", inputSchema: { type: "object", properties: { id: idProperty }, required: ["id"] } },
  { name: "type", description: "Click a text field, replace its text, and press Enter if submit is true.",
    inputSchema: { type: "object", properties: { id: idProperty, text: { type: "string" }, submit: { type: "boolean" } }, required: ["id", "text"] } },
  { name: "select", description: "Choose an option of a <select> element by its visible text or value.",
    inputSchema: { type: "object", properties: { id: idProperty, option: { type: "string" } }, required: ["id", "option"] } },
  { name: "press", description: "Press a key on the focused element.",
    inputSchema: { type: "object", properties: { key: { type: "string", enum: KEY_NAMES } }, required: ["key"] } },
  { name: "back", description: "Go back one page.", inputSchema: { type: "object", properties: {} } },
];

let browser = null, ranker = null;
const getBrowser = async () => (browser ??= await Browser.start({ headless: options.headless, position: options.position, size: options.size }));
const getRanker = async () => (ranker ??= await (await import("./src/ranker.mjs")).loadRanker());
const page = ({ url, title }) => `Page: ${title || "(untitled)"}\nURL: ${url}`;
const after = (verb, result) => [
  `${verb}.`, result.newTab ? "It opened a new tab, which is now the current tab." : "",
  result.covered ? `Note: another element was on top at that point (${result.covered}); the click went to it.` : "",
  page(result), "Call elements to see the page.",
].filter(Boolean).join("\n");

function checkId(id) {
  if (!/^\d+$/.test(String(id ?? ""))) throw new Error(`id must be a number from the elements list, got ${JSON.stringify(id)}`);
  return String(id);
}

async function listElements(args, record) {
  const b = await getBrowser();
  const snap = await b.snapshot(LIMIT);
  const total = snap.candidates.length;
  const count = total.toLocaleString("en-US");
  Object.assign(record, { elements_total: total, capped: snap.capped });
  if (MODE === "full") {
    record.elements_shown = total;
    const header = snap.capped ? `The first ${count} elements, in page order (the page has more; the list stops at ${count}):` : `All ${count} elements, in page order:`;
    return [page(snap), header, ...snap.candidates.map(([id, text]) => `[${id}] ${text}`)].join("\n");
  }
  const goal = String(args.goal ?? "").trim();
  if (!goal) throw new Error("goal is required");
  // A k below the default saves a few hundred tokens but a miss costs a whole extra model turn.
  const k = Math.max(K, Math.min(MAX_K, Number(args.k) || K));
  const start = performance.now();
  const ranked = await (await getRanker()).rank(goal, snap.candidates.map(([id, text]) => ({ id, text })), { k });
  record.rank_ms = Math.round(performance.now() - start);
  record.elements_shown = ranked.length;
  return [page(snap), `The ${ranked.length} most relevant of ${snap.capped ? `the first ${count}` : count} elements for "${goal}", best first:`,
    ...ranked.map(({ id, text }) => `[${id}] ${text}`)].join("\n");
}

async function callTool(name, args = {}, record) {
  const b = name === "elements" ? null : await getBrowser();
  switch (name) {
    case "open": {
      const url = String(args.url ?? "");
      return `Opened.\n${page(await b.open(/^[a-z]+:/i.test(url) ? url : `https://${url}`))}\nCall elements to see the page.`;
    }
    case "elements": return listElements(args, record);
    case "click": return after(`Clicked [${checkId(args.id)}]`, await b.click(args.id));
    case "type": return after(`Typed into [${checkId(args.id)}]${args.submit ? " and pressed Enter" : ""}`, await b.type(args.id, String(args.text ?? ""), Boolean(args.submit)));
    case "select": {
      const result = await b.select(checkId(args.id), args.option ?? "");
      return after(`Selected "${result.selected}" in [${args.id}]`, result);
    }
    case "press": return after(`Pressed ${args.key}`, await b.press(String(args.key)));
    case "back": return `Went back.\n${page(await b.back())}\nCall elements to see the page.`;
    default: throw new Error(`unknown tool ${name}`);
  }
}

const write = (message) => process.stdout.write(JSON.stringify(message) + "\n");
const log = (entry) => { if (options.log) appendFileSync(options.log, JSON.stringify(entry) + "\n"); };

async function handle(message) {
  const { id, method, params = {} } = message;
  if (id === undefined) return;
  if (method === "initialize") {
    return write({ jsonrpc: "2.0", id, result: {
      protocolVersion: params.protocolVersion ?? "2025-06-18",
      capabilities: { tools: {} },
      serverInfo: { name: `spotlit-browser-${MODE}`, version: "1.0.0" },
    } });
  }
  if (method === "ping") return write({ jsonrpc: "2.0", id, result: {} });
  if (method === "tools/list") {
    write({ jsonrpc: "2.0", id, result: { tools: TOOLS } });
    if (options.ready) writeFileSync(options.ready, "");
    return;
  }
  if (method !== "tools/call") return write({ jsonrpc: "2.0", id, error: { code: -32601, message: `unknown method ${method}` } });
  const record = { t: new Date().toISOString(), mode: MODE, tool: params.name, args: params.arguments ?? {} };
  const start = performance.now();
  let text, isError = false;
  try {
    text = await callTool(params.name, params.arguments ?? {}, record);
  } catch (error) {
    isError = true;
    text = error instanceof StaleElementError ? error.message : `Error: ${error.message}`;
    if (error.message === "tab closed") {
      browser?.close();
      browser = null;
      text = "Error: the browser window was closed. Call open with the start page to continue in a new window.";
    }
  }
  Object.assign(record, { ok: !isError, ms: Math.round(performance.now() - start), chars: text.length });
  if (browser) Object.assign(record, await browser.where());
  log(record);
  write({ jsonrpc: "2.0", id, result: { content: [{ type: "text", text }], isError } });
}

const shutdown = () => { browser?.close(); process.exit(0); };
process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);
const lines = createInterface({ input: process.stdin });
let queue = Promise.resolve();
lines.on("line", (line) => {
  if (!line.trim()) return;
  let message;
  try { message = JSON.parse(line); } catch { return; }
  queue = queue.then(() => handle(message)).catch((error) => process.stderr.write(`spotlit-browser: ${error.stack}\n`));
});
lines.on("close", () => queue.then(shutdown));
