// Spotlit, an element ranker for browser agents: tinylab stage 1 plus a BERT-tiny re-ranker (ADR-0012 to 0014).

import { Student, cosine } from "./tinylab.js";
import { WordPiece } from "./wordpiece.js";

export const DEPTH = 400;
const KEY_ATTRIBUTES = [
  ["aria-label"], ["title"], ["alt"], ["placeholder"], ["name"], ["type"], ["role"], ["value"],
];
const OWN_WORDS = 24, PARENT_WORDS = 12, ATTRIBUTE_CHARS = 60;
const SKIP_TAGS = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "TEMPLATE", "SVG", "PATH"]);
const CANDIDATE_SELECTOR = [
  "a", "button", "input", "select", "textarea", "option", "label", "summary", "[role]", "[onclick]",
  "[tabindex]", "[aria-label]", "[title]", "h1", "h2", "h3", "h4", "li", "td", "span", "div",
].join(",");

const words = (text) => text.split(/\s+/).filter(Boolean);

/** Query text in the training format: the goal plus up to three previous actions. */
export function queryOf(goal, history = []) {
  const recent = history.slice(-3);
  return goal + (recent.length ? ` Previous actions: ${recent.join("; ")}` : "");
}

/** A live element described like Mind2Web's cleaned HTML: tag, key attributes, own text, parent text. */
export function describeElement(element) {
  const parts = [element.tagName.toLowerCase()];
  for (const [name] of KEY_ATTRIBUTES) {
    const value = element.getAttribute?.(name);
    if (value) parts.push(value.slice(0, ATTRIBUTE_CHARS));
  }
  if ("value" in element && typeof element.value === "string" && element.value && !element.getAttribute("value")) {
    parts.push(element.value.slice(0, ATTRIBUTE_CHARS));
  }
  parts.push(...words(element.textContent ?? "").slice(0, OWN_WORDS));
  const parentWords = words(element.parentElement?.textContent ?? "").slice(0, PARENT_WORDS);
  if (parentWords.length) parts.push("|", ...parentWords);
  return parts.join(" ");
}

/** Elements an agent might act on or read, each with a stable id and its description. */
export function collectCandidates(root = document, limit = 2000) {
  const out = [];
  for (const element of root.querySelectorAll(CANDIDATE_SELECTOR)) {
    if (SKIP_TAGS.has(element.tagName.toUpperCase()) || element.closest("script,style,noscript,template")) continue;
    const text = describeElement(element);
    if (words(text).length < 2) continue;
    out.push({ id: String(out.length), element, text });
    if (out.length >= limit) break;
  }
  return out;
}

export class Spotlit {
  constructor(ort, stage1, session, tokenizer) {
    Object.assign(this, { ort, stage1, session, tokenizer });
  }

  /**
   * `ort` is ONNX Runtime Web. `stage1` and `reranker` are URLs or ArrayBuffers; `vocab` is a URL, or pass
   * `vocabText`. INT8 re-rankers must run on the "wasm" provider: WebGPU scores them wrongly (ORT Web 1.30).
   */
  static async load(ort, { stage1, reranker, vocab, vocabText }, executionProviders = ["wasm"]) {
    const bytes = async (source) => (source instanceof ArrayBuffer ? source : (await fetch(source)).arrayBuffer());
    const student = Student.fromBuffer(await bytes(stage1));
    const model = new Uint8Array(await bytes(reranker));
    const session = await ort.InferenceSession.create(model, { executionProviders, graphOptimizationLevel: "all" });
    const vocabulary = vocabText ?? (await (await fetch(vocab)).text());
    return new Spotlit(ort, student, session, new WordPiece(vocabulary));
  }

  /** Stage 1 order of every candidate, best first. */
  stage1Order(query, candidates) {
    const q = this.stage1.embed(query);
    return candidates
      .map((candidate) => [candidate, cosine(q, this.stage1.embed(candidate.text))])
      .sort((a, b) => b[1] - a[1] || (a[0].id < b[0].id ? -1 : a[0].id > b[0].id ? 1 : 0))
      .map(([candidate]) => candidate);
  }

  async scorePairs(query, texts) {
    const encoded = texts.map((text) => this.tokenizer.encodePair(query, text));
    const width = Math.max(...encoded.map((e) => e.inputIds.length));
    const size = texts.length * width;
    const ids = new BigInt64Array(size), mask = new BigInt64Array(size), types = new BigInt64Array(size);
    encoded.forEach((e, row) => e.inputIds.forEach((id, col) => {
      const at = row * width + col;
      ids[at] = BigInt(id); mask[at] = 1n; types[at] = BigInt(e.tokenTypeIds[col]);
    }));
    const dims = [texts.length, width];
    const feeds = {
      input_ids: new this.ort.Tensor("int64", ids, dims),
      attention_mask: new this.ort.Tensor("int64", mask, dims),
      token_type_ids: new this.ort.Tensor("int64", types, dims),
    };
    return (await this.session.run(feeds)).score.data;
  }

  /** Candidates (`{id, text}`) for `goal`, best first, each with its stage 1 rank; returns the top `k`. */
  async rank(goal, candidates, { k = 50, depth = DEPTH, history = [] } = {}) {
    if (!candidates.length) return [];
    const query = queryOf(goal, history);
    const order = this.stage1Order(query, candidates);
    const head = order.slice(0, depth);
    const scores = await this.scorePairs(query, head.map((c) => c.text));
    const reranked = head.map((candidate, i) => ({ ...candidate, score: scores[i], stage1Rank: i + 1 }))
      .sort((a, b) => b.score - a.score || a.stage1Rank - b.stage1Rank);
    const rest = order.slice(depth).map((candidate, i) => ({ ...candidate, score: -Infinity, stage1Rank: depth + i + 1 }));
    return [...reranked, ...rest].slice(0, k);
  }
}
