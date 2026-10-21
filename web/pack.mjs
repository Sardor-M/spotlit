#!/usr/bin/env node
/** Packs models/spotlit-v1 into one gzipped file, dist/spotlit-v1.weights.gz, for spotlit.loadPacked(). */
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { gzipSync } from "node:zlib";

const ROOT = new URL("../", import.meta.url).pathname;
const MODEL = `${ROOT}models/spotlit-v1/`;
const FILES = ["stage1.tlab", "reranker.int8.onnx", "vocab.txt"];

export function pack() {
  const manifest = JSON.parse(readFileSync(`${MODEL}manifest.json`, "utf8"));
  const parts = FILES.map((name) => readFileSync(MODEL + name));
  let offset = 0;
  const files = FILES.map((name, i) => {
    const sha256 = createHash("sha256").update(parts[i]).digest("hex");
    if (sha256 !== manifest.files[name].sha256) throw new Error(`${name} does not match manifest.json`);
    const entry = { name, offset, length: parts[i].length, sha256 };
    offset += parts[i].length;
    return entry;
  });
  const { name, version, license, task, query, element, operating_point, stage1, stage2 } = manifest;
  const header = Buffer.from(JSON.stringify({ name, version, license, task, query, element, operating_point, stage1, stage2, files }));
  const headerLength = Buffer.alloc(4);
  headerLength.writeUInt32LE(header.length);
  const packed = gzipSync(Buffer.concat([Buffer.from("SPOTLIT1"), headerLength, header, ...parts]), { level: 9 });
  const path = `${ROOT}dist/spotlit-v1.weights.gz`;
  mkdirSync(`${ROOT}dist`, { recursive: true });
  writeFileSync(path, packed);
  return { path, bytes: packed.length, sha256: createHash("sha256").update(packed).digest("hex") };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const { path, bytes, sha256 } = pack();
  console.log(`${path}: ${(bytes / 1e6).toFixed(2)} MB, sha256 ${sha256}`);
}
