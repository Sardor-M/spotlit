import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const CHROME = process.env.CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
export const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** Starts Chrome on a throwaway profile; returns its DevTools port and a close function. */
export async function launchChrome({ headless = false, position = "0,0", size = "1280,900" } = {}) {
  const profile = mkdtempSync(join(tmpdir(), "spotlit-chrome-"));
  const child = spawn(CHROME, [
    `--user-data-dir=${profile}`, "--remote-debugging-port=0", "--no-first-run", "--no-default-browser-check",
    "--disable-search-engine-choice-screen", "--disable-features=Translate", `--window-position=${position}`,
    `--window-size=${size}`, ...(headless ? ["--headless=new"] : []), "about:blank",
  ], { stdio: "ignore" });
  const portFile = join(profile, "DevToolsActivePort");
  let port = 0;
  for (let i = 0; i < 200 && !port; i++) {
    if (existsSync(portFile)) port = Number(readFileSync(portFile, "utf8").split("\n")[0]) || 0;
    if (!port) await sleep(100);
  }
  if (!port) throw new Error(`Chrome did not start (${CHROME}); set CHROME to its binary`);
  const close = () => {
    child.kill();
    setTimeout(() => rmSync(profile, { recursive: true, force: true }), 300).unref();
  };
  return { port, close };
}

export const listPages = async (port) =>
  (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).filter((target) => target.type === "page");

export const browserVersion = async (port) => (await fetch(`http://127.0.0.1:${port}/json/version`)).json();

/** A DevTools Protocol session on one page target. */
export async function connectTab(target) {
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    ws.addEventListener("open", resolve, { once: true });
    ws.addEventListener("error", () => reject(new Error("DevTools connection failed")), { once: true });
  });
  let next = 0;
  const pending = new Map(), listeners = new Map();
  ws.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      const { resolve, reject } = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) reject(new Error(message.error.message)); else resolve(message.result);
    } else for (const listener of listeners.get(message.method) ?? []) listener(message.params);
  });
  ws.addEventListener("close", () => {
    for (const { reject } of pending.values()) reject(new Error("tab closed"));
    pending.clear();
  });
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    if (ws.readyState !== WebSocket.OPEN) return reject(new Error("tab closed"));
    const id = ++next;
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async (expression) => {
    const result = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description ?? result.exceptionDetails.text);
    return result.result.value;
  };
  const on = (method, listener) => listeners.set(method, [...(listeners.get(method) ?? []), listener]);
  return { id: target.id, send, evaluate, on, close: () => ws.close() };
}
