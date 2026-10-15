# Spotlit v1

A 6.7 MB open-weight model that ranks a web page's elements against an AI agent's goal, in
the browser. An agent sends only the top 50 elements to its large model instead of the whole
page: 32-43x fewer tokens per step, with the element a person actually clicked kept in the
top 50 for 76-81% of steps on unseen Mind2Web test sites.

## Files

| File | Bytes | What it is |
| :--- | ---: | :--- |
| `stage1.tlab` | 1,990,696 | Stage 1: hashed byte n-gram table, 6,144 x 320, INT8, no tokenizer |
| `reranker.int8.onnx` | 4,488,511 | Stage 2: BERT-tiny cross-encoder (4.4M params), INT8, ONNX |
| `vocab.txt` | 231,508 | Stage 2's WordPiece vocabulary |
| `manifest.json` | | checksums, input formats, operating point |

## How it works

1. **Describe.** Each element becomes one line: tag, then labels (aria-label, title, alt,
   placeholder, name, type, role, value; 60 characters each), then up to 24 words of its own
   text, then `|` and up to 12 words of its parent's text.
2. **Stage 1** embeds the goal and every element with the hashed n-gram table and ranks them
   by cosine similarity. About 35 ms for 500 elements.
3. **Stage 2** reads `[CLS] goal [SEP] element [SEP]` (128 tokens) for stage 1's top 400 and
   re-scores them. About 185 ms with eight WebAssembly threads.
4. The agent receives the top 50.

The goal is the task plus up to three previous actions: `goal + " Previous actions: " + a; b; c`.

## Use

```js
import * as ort from "onnxruntime-web";
import { Spotlit, collectCandidates } from "./web/src/spotlit.js";

const ranker = await Spotlit.load(ort, {
  stage1: "models/spotlit-v1/stage1.tlab",
  reranker: "models/spotlit-v1/reranker.int8.onnx",
  vocab: "models/spotlit-v1/vocab.txt",
});
const top50 = await ranker.rank("Find the retired players named James Smith", collectCandidates(document));
```

Run the INT8 re-ranker on ONNX Runtime Web's `wasm` provider. On the WebGPU provider of ORT
Web 1.30 the INT8 graph returns wrong scores. WebAssembly threads need a cross-origin
isolated page (COOP/COEP headers); without them it runs on one thread at about 1 s per step.

## Evaluation

Mind2Web, share of steps whose clicked element is in the top k. Test splits were run once,
after the files were frozen (sha256 in `manifest.json`).

| Split | Steps | BM25 r@50 | Stage 1 r@50 | **r@10** | **r@50** | **r@100** | Tokens, top 50 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dev (held-out train websites) | 1,106 | 38.2% | 55.2% | 49.3% | 73.1% | 80.1% | 46x fewer |
| Test, cross-task | 2,094 | 44.2% | 69.2% | 57.2% | 80.8% | 85.8% | 43x fewer |
| Test, cross-website | 1,373 | 43.8% | 56.7% | 48.7% | 75.8% | 82.3% | 34x fewer |
| Test, cross-domain | 5,911 | 44.5% | 57.7% | 51.5% | 75.9% | 83.6% | 32x fewer |

For reference, MindAct's DeBERTa-v3-base ranker (the benchmark authors'; 184M parameters
with its embedding table, about 29 times ours) reports 88.9 / 85.3 / 85.7% r@50 on the same
test splits. Candidate descriptions differ, and steps whose clicked element is missing from
the page HTML (4-7%) count as misses here, so the comparison is approximate. Full details:
`docs/benchmarks/spotlit.md`.

## Limits

- Trained on Mind2Web's English websites from 2023; other languages and very different
  page structures are untested.
- About one step in four loses the clicked element from the top 50. An agent should be able
  to ask for more elements or fall back to the full page.
- The token counts are estimates (four characters per token) against the step's cleaned HTML.

## License

Apache-2.0 (`LICENSE`). Credits for the base model and training data are in `NOTICE`.
