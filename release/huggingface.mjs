#!/usr/bin/env node
/** Assembles dist/hf/spotlit-v1/, the full open-weight release for the Hugging Face Hub. HF_REPO sets the repo id. */
import { createHash } from "node:crypto";
import { copyFileSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { pack } from "../web/pack.mjs";

const ROOT = new URL("../", import.meta.url).pathname;
const MODEL = join(ROOT, "models/spotlit-v1");
const CHECKPOINT = join(ROOT, "checkpoints/reranker-bert_uncased_L-2_H-128_A-2-v1");
const OUT = join(ROOT, "dist/hf/spotlit-v1");
const REPO = process.env.HF_REPO ?? "Sardori/spotlit-v1";
const GITHUB = "https://github.com/Sardor-M/spotlit";

const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const write = (path, data) => { mkdirSync(dirname(join(OUT, path)), { recursive: true }); writeFileSync(join(OUT, path), data); };
const copy = (from, to) => write(to, readFileSync(from));

/** The .tlab table and scales as a safetensors file: same numbers, a standard container. */
function tlabToSafetensors(buffer) {
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  const buckets = view.getUint32(12, true), dims = view.getUint32(16, true), sizes = view.getUint32(24, true);
  const scalesStart = 28 + 4 * sizes, tableStart = scalesStart + 4 * buckets;
  const scales = buffer.subarray(scalesStart, tableStart), table = buffer.subarray(tableStart, tableStart + buckets * dims);
  const header = {
    __metadata__: { format: "pt", source: "stage1.tlab", note: "row r of the embedding is scales[r] * table[r]; see config.json for the encoder" },
    scales: { dtype: "F32", shape: [buckets], data_offsets: [0, scales.length] },
    table: { dtype: "I8", shape: [buckets, dims], data_offsets: [scales.length, scales.length + table.length] },
  };
  let json = JSON.stringify(header);
  json += " ".repeat((8 - ((8 + json.length) % 8)) % 8);
  const size = Buffer.alloc(8);
  size.writeBigUInt64LE(BigInt(json.length));
  return Buffer.concat([size, Buffer.from(json), scales, table]);
}

function stage1Config(buffer) {
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  const sizes = view.getUint32(24, true);
  return {
    format: "tlab", format_version: view.getUint32(4, true), encoder_version: view.getUint32(8, true),
    num_buckets: view.getUint32(12, true), dims: view.getUint32(16, true), max_input_bytes: view.getUint32(20, true),
    ngram_sizes: Array.from({ length: sizes }, (_, i) => view.getUint32(28 + 4 * i, true)),
    normalize: "Take the UTF-8 bytes. Lower-case ASCII letters. Keep ASCII letters, digits and bytes 0x80 and above; every other byte becomes a space. Collapse runs of spaces, drop leading and trailing spaces, cut to max_input_bytes, then add one space at each end.",
    hash: "64-bit FNV-1a (offset basis 0xcbf29ce484222325, prime 0x100000001b3) over each n-gram's bytes, then XOR in the n-gram length and multiply by the prime once more.",
    bucket: "the full 64-bit hash modulo num_buckets",
    sign: "-1 if bit 63 of the hash is set, otherwise +1",
    embedding: "Sum sign * scales[bucket] * table[bucket] over every n-gram of every size, then L2-normalize.",
    score: "cosine similarity of the goal's and the element's embeddings",
    reference: "web/tinylab.js",
  };
}

export function buildRelease() {
  rmSync(OUT, { recursive: true, force: true });
  const tlab = readFileSync(join(MODEL, "stage1.tlab"));
  copy(join(MODEL, "stage1.tlab"), "stage1/stage1.tlab");
  write("stage1/model.safetensors", tlabToSafetensors(tlab));
  write("stage1/config.json", `${JSON.stringify(stage1Config(tlab), null, 2)}\n`);

  for (const file of ["config.json", "model.safetensors", "tokenizer.json"]) copy(join(CHECKPOINT, file), `reranker/${file}`);
  const tokenizerConfig = JSON.parse(readFileSync(join(CHECKPOINT, "tokenizer_config.json"), "utf8"));
  write("reranker/tokenizer_config.json", `${JSON.stringify({ ...tokenizerConfig, model_max_length: 128 }, null, 2)}\n`);
  copy(join(MODEL, "vocab.txt"), "reranker/vocab.txt");
  copy(join(MODEL, "reranker.int8.onnx"), "onnx/reranker.int8.onnx");

  copy(pack().path, "web/spotlit-v1.weights.gz");
  for (const file of ["spotlit.js", "tinylab.js", "wordpiece.js"]) copy(join(ROOT, "web/src", file), `web/${file}`);
  copy(join(ROOT, "LICENSE"), "web/LICENSE");

  copy(join(MODEL, "LICENSE"), "LICENSE");
  write("NOTICE", readFileSync(join(MODEL, "NOTICE"), "utf8")
    .replace("reranker.int8.onnx\n", "reranker/ and onnx/reranker.int8.onnx\n")
    .replace("stage1.tlab was initialized", "stage1/ was initialized")
    .concat("\nThe runtime in web/ is MIT licensed (web/LICENSE).\n"));
  write("README.md", readFileSync(join(ROOT, "release/model-card.md"), "utf8").replaceAll("{{REPO}}", REPO).replaceAll("{{GITHUB}}", GITHUB));

  const shipped = JSON.parse(readFileSync(join(MODEL, "manifest.json"), "utf8"));
  // The Hub counts a repo's downloads by requests for its root config.json.
  write("config.json", `${JSON.stringify({
    name: shipped.name, version: shipped.version, task: shipped.task,
    stage1: { file: "stage1/stage1.tlab", encoder: "stage1/config.json", safetensors: "stage1/model.safetensors" },
    stage2: { onnx: "onnx/reranker.int8.onnx", full_precision: "reranker/", base: shipped.stage2.base, max_tokens: shipped.stage2.max_tokens },
    web: "web/spotlit-v1.weights.gz",
    operating_point: shipped.operating_point,
  }, null, 2)}\n`);
  const files = {};
  const walk = (dir) => readdirSync(dir).sort().forEach((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return walk(path);
    const bytes = readFileSync(path);
    files[relative(OUT, path)] = { bytes: bytes.length, sha256: sha256(bytes) };
  });
  walk(OUT);
  const { files: _, ...about } = shipped;
  write("manifest.json", `${JSON.stringify({ ...about, repo: REPO, files }, null, 2)}\n`);
  return { out: OUT, files };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const { out, files } = buildRelease();
  const total = Object.values(files).reduce((sum, f) => sum + f.bytes, 0);
  for (const [path, { bytes }] of Object.entries(files)) console.log(`${(bytes / 1e6).toFixed(2).padStart(7)} MB  ${path}`);
  console.log(`${(total / 1e6).toFixed(1)} MB in ${Object.keys(files).length} files, plus manifest.json -> ${out}`);
}
