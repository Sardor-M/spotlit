// Parity and speed of the JS runtime: node web/test/parity.mjs [model dir]
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { Student, features } from "../src/tinylab.js";

const root = new URL("../../", import.meta.url);
let failures = 0;

function checkEncoderGoldens() {
  const lines = readFileSync(new URL("tests/encoder_goldens.txt", root), "utf8").trim().split("\n");
  const spec = { numBuckets: 0, ngramSizes: [], maxInputBytes: 0 };
  const header = /buckets=(\d+) ngrams=([\d,]+) max_input_bytes=(\d+)/.exec(lines[1]);
  spec.numBuckets = Number(header[1]);
  spec.ngramSizes = header[2].split(",").map(Number);
  spec.maxInputBytes = Number(header[3]);
  let cases = 0;
  for (const line of lines.slice(2)) {
    const [, name, hex, , ...pairs] = line.split(" ");
    const text = hex === "-" ? "" : new TextDecoder().decode(Buffer.from(hex, "hex"));
    const got = features(text, spec).map(([b, c]) => `${b}:${c}`).join(" ");
    if (got !== pairs.join(" ")) {
      failures++;
      console.log(`  FAIL encoder case ${name}`);
    }
    cases++;
  }
  console.log(`encoder goldens: ${cases - failures}/${cases} cases match the Python encoder`);
}

function checkModel(dir) {
  const blob = readFileSync(new URL("student.tlab", dir));
  const goldens = JSON.parse(readFileSync(new URL("goldens.json", dir), "utf8"));
  const sha = createHash("sha256").update(blob).digest("hex");
  if (sha !== goldens.sha256) {
    failures++;
    console.log("  FAIL goldens were made for another model file");
    return;
  }
  const buffer = blob.buffer.slice(blob.byteOffset, blob.byteOffset + blob.byteLength);
  const student = Student.fromBuffer(buffer);
  let worst = 0;
  for (const { name, text, vector } of goldens.cases) {
    const got = student.embed(text);
    const diff = Math.max(...vector.map((x, d) => Math.abs(x - got[d])));
    worst = Math.max(worst, diff);
    if (diff > 1e-6) {
      failures++;
      console.log(`  FAIL embedding ${name}: max diff ${diff.toExponential(2)}`);
    }
  }
  console.log(`embedding goldens: ${goldens.cases.length} texts, max abs diff ${worst.toExponential(2)}`);

  const queries = goldens.cases.map((c) => c.text).filter((t) => t.length > 0 && t.length < 80);
  const passage = "Wrap many inserts in a single transaction and index after loading. ".repeat(10).slice(0, 600);
  for (const [label, texts, rounds] of [["query", queries, 200], ["512-byte passage", [passage], 2000]]) {
    for (const text of texts) student.embed(text);
    const times = [];
    for (let round = 0; round < rounds; round++) {
      for (const text of texts) {
        const start = performance.now();
        student.embed(text);
        times.push(performance.now() - start);
      }
    }
    times.sort((a, b) => a - b);
    const mean = times.reduce((a, b) => a + b, 0) / times.length;
    console.log(
      `${label} embedding: mean ${mean.toFixed(3)} ms, p95 ${times[Math.floor(times.length * 0.95)].toFixed(3)} ms` +
        ` over ${times.length} runs (node ${process.version}, one thread)`,
    );
  }
}

checkEncoderGoldens();
const modelDir = new URL(`${process.argv[2] ?? "dist/tinylab-student-v1"}/`, root);
if (existsSync(new URL("student.tlab", modelDir))) checkModel(modelDir);
else console.log(`no model at ${modelDir.pathname}; run tinylab-export to check embeddings`);
process.exit(failures ? 1 : 0);
