import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-web/wasm";
import * as spotlit from "../../web/src/spotlit.js";

const MODEL_DIR = fileURLToPath(new URL("../../models/spotlit-v1/", import.meta.url));
const buffer = (path) => { const b = readFileSync(path); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };

/** spotlit v1 on ONNX Runtime Web's WebAssembly backend, the same engine the browser runs. */
export async function loadRanker({ modelDir = MODEL_DIR } = {}) {
  ort.env.wasm.numThreads = 1; // ORT Web has no WebAssembly threads under Node; Chrome with 8 threads is about 5x faster
  ort.env.logLevel = "error";
  return spotlit.load(ort, {
    stage1: buffer(modelDir + "stage1.tlab"),
    reranker: buffer(modelDir + "reranker.int8.onnx"),
    vocabText: readFileSync(modelDir + "vocab.txt", "utf8"),
  });
}
