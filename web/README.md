# web

The browser runtime: `src/tinylab.js`, one ES module with no dependencies. It reads a
`.tlab` file (`python/tinylab/export/README.md`) and embeds text on the calling thread.

```js
import { Student, cosine } from "./src/tinylab.js";

const student = Student.fromBuffer(await (await fetch("student.tlab")).arrayBuffer());
const score = cosine(student.embed("reciprocal rank fusion"), student.embed(passage));
```

Vectors are L2-normalized, so cosine is a dot product. Every vector shares a large first
component, so raw cosines sit high (unrelated texts score about 0.6); use them to rank, not
as calibrated similarities.

## Checks

```sh
uv run tinylab-export                 # writes dist/tinylab-student-v1
node web/test/parity.mjs              # encoder goldens, embedding goldens, timings
python3 -m http.server 8765 --bind 127.0.0.1   # then open /web/bench.html
```

`parity.mjs` checks the encoder against `tests/encoder_goldens.txt` and the embeddings
against the exporter's `goldens.json`. `bench.html` repeats both in a browser and times
embedding on the main thread.

## Element ranker (ADR-0012 to 0014)

Two stages for browser agents: stage 1 is the same `.tlab` runtime on a table trained on
Mind2Web; stage 2 re-scores stage 1's top 400 elements with a BERT-tiny cross-encoder through
ONNX Runtime Web, INT8 on WebAssembly (INT8 on WebGPU gives wrong scores in ORT Web 1.30).
`src/wordpiece.js` is the dependency-free tokenizer for stage 2, exact against Hugging Face's.

```sh
uv run tinylab-export --checkpoint checkpoints/ranker-linear-6144x320-v1 --out dist/tinylab-ranker-v1
uv run tinylab-export-reranker --reranker checkpoints/reranker-bert_uncased_L-2_H-128_A-2-v1
node web/test/wordpiece-parity.mjs dist/reranker-bert_uncased_L-2_H-128_A-2-v1
```

`bench-rerank.html` times both stages per page step; it needs a cross-origin isolated server
(COOP/COEP headers) for WebAssembly threads and ONNX Runtime Web served at `?ort=` (default
`/ortweb/`).
