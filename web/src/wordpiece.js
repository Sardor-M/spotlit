// BERT uncased WordPiece for the stage 2 re-ranker, matching Hugging Face's fast tokenizer exactly.

const CONTROL = /\p{C}/u;
const WHITESPACE = /\p{White_Space}/u;
const PUNCTUATION = /\p{P}/u;
const NONSPACING_MARK = /\p{Mn}/u;
const ASCII_PUNCTUATION = /[!-\/:-@\[-`{-~]/;
const MAX_WORD_CHARS = 100;
const CJK_RANGES = [
  [0x4e00, 0x9fff], [0x3400, 0x4dbf], [0x20000, 0x2a6df], [0x2a700, 0x2b73f],
  [0x2b740, 0x2b81f], [0x2b820, 0x2ceaf], [0xf900, 0xfaff], [0x2f800, 0x2fa1f],
];

const isCjk = (cp) => CJK_RANGES.some(([lo, hi]) => cp >= lo && cp <= hi);
const isWhitespace = (c) => c === "\t" || c === "\n" || c === "\r" || WHITESPACE.test(c);
const isControl = (c) => c !== "\t" && c !== "\n" && c !== "\r" && CONTROL.test(c);
const isPunctuation = (c) => ASCII_PUNCTUATION.test(c) || PUNCTUATION.test(c);

/** BertNormalizer: clean text, space out CJK, strip accents, lowercase (in that order). */
export function normalize(text) {
  let out = "";
  for (const c of text) {
    if (c === "\0" || c === "�" || isControl(c)) continue;
    if (isWhitespace(c)) out += " ";
    else if (isCjk(c.codePointAt(0))) out += ` ${c} `;
    else out += c;
  }
  let stripped = "";
  for (const c of out.normalize("NFD")) if (!NONSPACING_MARK.test(c)) stripped += c;
  return stripped.toLowerCase();
}

/** BertPreTokenizer: split on whitespace, and isolate every punctuation character. */
export function preTokenize(text) {
  const words = [];
  let current = "";
  for (const c of text) {
    if (isWhitespace(c)) {
      if (current) words.push(current);
      current = "";
    } else if (isPunctuation(c)) {
      if (current) words.push(current);
      words.push(c);
      current = "";
    } else {
      current += c;
    }
  }
  if (current) words.push(current);
  return words;
}

export class WordPiece {
  constructor(vocabText, maxLength = 128) {
    this.vocab = new Map(vocabText.split("\n").filter((t) => t.length).map((token, id) => [token, id]));
    this.maxLength = maxLength;
    [this.cls, this.sep, this.unk, this.pad] = ["[CLS]", "[SEP]", "[UNK]", "[PAD]"].map((t) => this.vocab.get(t));
  }

  wordIds(word) {
    const chars = [...word];
    if (chars.length > MAX_WORD_CHARS) return [this.unk];
    const ids = [];
    for (let start = 0; start < chars.length; ) {
      let end = chars.length, id;
      for (; end > start; end--) {
        id = this.vocab.get((start > 0 ? "##" : "") + chars.slice(start, end).join(""));
        if (id !== undefined) break;
      }
      if (end === start) return [this.unk];
      ids.push(id);
      start = end;
    }
    return ids;
  }

  encode(text) {
    return preTokenize(normalize(text)).flatMap((word) => this.wordIds(word));
  }

  /** [CLS] a [SEP] b [SEP], truncated longest-first as the Rust tokenizer does; an empty b means no pair. */
  encodePair(a, b) {
    let first = this.encode(a);
    if (b === "") {
      first = first.slice(0, this.maxLength - 2);
      return { inputIds: [this.cls, ...first, this.sep], tokenTypeIds: new Array(first.length + 2).fill(0) };
    }
    let second = this.encode(b);
    const budget = this.maxLength - 3;
    if (first.length + second.length > budget) {
      // Fitted to the Rust tokenizer's longest-first rule on 1,849 length pairs (ADR-0014).
      const n1 = first.length, n2 = second.length, half = Math.floor(budget / 2), shorter = Math.min(n1, n2);
      let keep1 = half, keep2 = budget - half;
      if (shorter <= half) [keep1, keep2] = n1 === shorter ? [n1, budget - n1] : [budget - n2, n2];
      else if (shorter < this.maxLength && n1 > n2) [keep1, keep2] = [budget - half, half];
      first = first.slice(0, keep1);
      second = second.slice(0, keep2);
    }
    const inputIds = [this.cls, ...first, this.sep, ...second, this.sep];
    const tokenTypeIds = inputIds.map((_, i) => (i < first.length + 2 ? 0 : 1));
    return { inputIds, tokenTypeIds };
  }
}
