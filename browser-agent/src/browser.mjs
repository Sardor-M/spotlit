import { readFileSync } from "node:fs";
import { browserVersion, connectTab, launchChrome, listPages, sleep } from "./chrome.mjs";

const SPOTLIT_SOURCE = readFileSync(new URL("../../web/src/spotlit.js", import.meta.url), "utf8")
  .replace(/^import .*$/gm, "").replace(/^export /gm, "");
const KEYS = {
  Enter: [13, "\r"], Escape: [27], Tab: [9], Backspace: [8], Space: [32, " "], ArrowDown: [40], ArrowUp: [38],
  ArrowLeft: [37], ArrowRight: [39], PageDown: [34], PageUp: [33], Home: [36], End: [35],
};
export const KEY_NAMES = Object.keys(KEYS);

// Runs spotlit's own collectCandidates in the page, keeps visible elements, and remembers them by id.
const snapshotExpression = (limit) => `(() => {
  const { collectCandidates } = (() => { ${SPOTLIT_SOURCE}
    return { collectCandidates }; })();
  const visible = (element) => {
    if (element.checkVisibility && !element.checkVisibility({ visibilityProperty: true })) return false;
    const box = element.getBoundingClientRect();
    return box.width > 0 && box.height > 0;
  };
  const elements = [], candidates = [];
  let capped = false;
  for (const candidate of collectCandidates(document, 1e6)) {
    if (!visible(candidate.element)) continue;
    if (elements.length >= ${limit}) { capped = true; break; }
    candidates.push([String(elements.length), candidate.text]);
    elements.push(candidate.element);
  }
  window.__spotlitAgent = { elements };
  return { url: location.href, title: document.title, candidates, capped };
})()`;

const elementExpression = (id, body) => `(() => {
  const element = window.__spotlitAgent?.elements[${Number(id)}];
  if (!element || !element.isConnected) return { stale: true };
  ${body}
})()`;

const POINT = `
  element.scrollIntoView({ block: "center", inline: "center", behavior: "instant" });
  const box = element.getBoundingClientRect();
  // A link that wraps has its box centre between the lines, so aim at the middle of a line that hits it.
  const points = [...element.getClientRects()].filter((r) => r.width > 0 && r.height > 0)
    .map((r) => [r.left + r.width / 2, r.top + r.height / 2]).concat([[box.left + box.width / 2, box.top + box.height / 2]]);
  const hits = ([px, py]) => { const at = document.elementFromPoint(px, py); return at && (at === element || element.contains(at)); };
  const [x, y] = points.find(hits) ?? points.at(-1);
  const top = document.elementFromPoint(x, y);
  const covered = top && top !== element && !element.contains(top)
    ? (top.tagName.toLowerCase() + " " + (top.textContent ?? "").trim().split(/\\s+/).slice(0, 8).join(" ")).trim() : null;
  return { x, y, covered };`;

export class StaleElementError extends Error {}

/** One Chrome window the agent drives by element id. */
export class Browser {
  static async start(options = {}) {
    const chrome = await launchChrome(options);
    const browser = new Browser(chrome, options);
    const [page] = await listPages(chrome.port);
    await browser.attach(page);
    return browser;
  }

  constructor(chrome, { headless = false } = {}) {
    Object.assign(this, { chrome, headless, tab: null, seen: new Set() });
  }

  async attach(target) {
    this.tab?.close();
    this.tab = await connectTab(target);
    this.seen.add(target.id);
    this.tab.on("Page.javascriptDialogOpening", ({ type }) =>
      this.tab.send("Page.handleJavaScriptDialog", { accept: type !== "confirm" }).catch(() => {}));
    await this.tab.send("Page.enable");
    await this.tab.send("Runtime.enable");
    if (this.headless) {
      const { "User-Agent": agent } = await browserVersion(this.chrome.port);
      await this.tab.send("Network.setUserAgentOverride", { userAgent: agent.replace("HeadlessChrome", "Chrome") });
    }
  }

  async settle({ min = 700, max = 10000 } = {}) {
    await sleep(min);
    const start = Date.now();
    let last = -1, stable = 0;
    while (Date.now() - start < max) {
      const state = await this.tab.evaluate(`[document.readyState, document.getElementsByTagName("*").length]`).catch(() => null);
      if (state?.[0] === "complete") {
        stable = state[1] === last ? stable + 1 : 0;
        last = state[1];
        if (stable >= 2) break;
      }
      await sleep(250);
    }
    const fresh = (await listPages(this.chrome.port)).filter((target) => !this.seen.has(target.id));
    if (fresh.length) {
      await this.attach(fresh.at(-1));
      await this.settle({ min: 300, max });
      return true;
    }
    return false;
  }

  async where() {
    return this.tab.evaluate(`({ url: location.href, title: document.title })`).catch(() => ({ url: "", title: "" }));
  }

  async open(url) {
    const { errorText } = await this.tab.send("Page.navigate", { url });
    if (errorText) throw new Error(`could not open ${url}: ${errorText}`);
    await this.settle();
    return this.where();
  }

  snapshot(limit = 2000) {
    return this.tab.evaluate(snapshotExpression(limit));
  }

  async inElement(id, body) {
    const result = await this.tab.evaluate(elementExpression(id, body));
    if (result?.stale) throw new StaleElementError(`element ${id} is not on the page any more; call elements again`);
    return result;
  }

  async mouseClick(x, y) {
    for (const type of ["mouseMoved", "mousePressed", "mouseReleased"]) {
      await this.tab.send("Input.dispatchMouseEvent", { type, x, y, button: "left", clickCount: type === "mouseMoved" ? 0 : 1 });
    }
  }

  async click(id) {
    const { x, y, covered } = await this.inElement(id, POINT);
    await this.mouseClick(x, y);
    const newTab = await this.settle();
    return { ...(await this.where()), covered, newTab };
  }

  async type(id, text, submit = false) {
    const { x, y, covered } = await this.inElement(id, POINT);
    await this.mouseClick(x, y);
    await sleep(150);
    await this.tab.evaluate(`(() => {
      const active = document.activeElement;
      if (active && typeof active.select === "function") active.select();
      else if (active?.isContentEditable) document.execCommand("selectAll");
      return active ? active.tagName.toLowerCase() : null;
    })()`);
    await this.tab.send("Input.insertText", { text });
    if (submit) await this.press("Enter", false);
    const newTab = await this.settle();
    return { ...(await this.where()), covered, newTab };
  }

  async select(id, option) {
    const result = await this.inElement(id, `
      let select = element.tagName === "SELECT" ? element : element.closest("select") ?? element.querySelector("select");
      if (!select) return { error: "element " + ${JSON.stringify(String(id))} + " is not a <select>; click it instead" };
      const want = ${JSON.stringify(String(option).trim().toLowerCase())};
      const options = [...select.options];
      const match = options.find((o) => o.text.trim().toLowerCase() === want || o.value.toLowerCase() === want)
        ?? options.find((o) => o.text.trim().toLowerCase().includes(want));
      if (!match) return { error: "no option matches; options are: " + options.map((o) => o.text.trim()).slice(0, 40).join(" | ") };
      select.value = match.value;
      select.dispatchEvent(new Event("input", { bubbles: true }));
      select.dispatchEvent(new Event("change", { bubbles: true }));
      return { selected: match.text.trim() };`);
    if (result.error) throw new Error(result.error);
    const newTab = await this.settle();
    return { ...(await this.where()), selected: result.selected, newTab };
  }

  async press(key, settle = true) {
    const [code, text] = KEYS[key] ?? [];
    if (!code) throw new Error(`unknown key ${key}; use one of ${KEY_NAMES.join(", ")}`);
    const common = { key: key === "Space" ? " " : key, code: key, windowsVirtualKeyCode: code };
    await this.tab.send("Input.dispatchKeyEvent", { type: text ? "keyDown" : "rawKeyDown", ...common, ...(text ? { text } : {}) });
    await this.tab.send("Input.dispatchKeyEvent", { type: "keyUp", ...common });
    if (!settle) return null;
    const newTab = await this.settle();
    return { ...(await this.where()), newTab };
  }

  async back() {
    await this.tab.evaluate("history.back()");
    await this.settle();
    return this.where();
  }

  close() {
    this.tab?.close();
    this.chrome.close();
  }
}
