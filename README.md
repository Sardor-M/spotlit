# tinylab

Open-weight models small enough to run in a browser tab.

**Element ranker v1** (6.7 MB) ranks a web page's elements against an AI agent's goal, so
the agent sends its large model the 50 elements that matter instead of the whole page:
32-43x fewer tokens per step, in about a quarter of a second, entirely in the browser.

```
   web page: 400-500 elements
        |
   describe     each element as one line: tag, labels, own text, parent text
        |
   stage 1      2.0 MB  hashed byte n-grams, no tokenizer        every element, ~35 ms
        |  top 400
   stage 2      4.5 MB  BERT-tiny cross-encoder, INT8 ONNX       400 pairs, ~185 ms
        |  top 50
   AI agent's large model  ->  click, type, select
```

## Results

Mind2Web test splits, run once after the model was frozen. Share of steps whose clicked
element is in the top 50.

| System | Size | Cross-task | Cross-website | Cross-domain |
| :--- | ---: | ---: | ---: | ---: |
| Keyword search (BM25) | 0 | 44.2% | 43.8% | 44.5% |
| tinylab stage 1 alone | 2.0 MB | 69.2% | 56.7% | 57.7% |
| **tinylab element ranker v1** | **6.7 MB** | **80.8%** | **75.8%** | **75.9%** |
| MindAct DeBERTa-v3-base (paper) | 184M params | 88.9% | 85.3% | 85.7% |

Tokens sent at the top 50: 43x / 34x / 32x fewer than the step's cleaned HTML. Speed is
Chrome on an M4 Pro with eight WebAssembly threads. Details and every variant tried are in
[docs/benchmarks](docs/benchmarks/README.md).

## Use it

The model is in [`models/element-ranker-v1`](models/element-ranker-v1/README.md) (also
attached to the GitHub release): `stage1.tlab`, `reranker.int8.onnx`, `vocab.txt`.

```js
import * as ort from "onnxruntime-web";
import { ElementRanker, collectCandidates } from "./web/src/element-ranker.js";

const ranker = await ElementRanker.load(ort, {
  stage1: "models/element-ranker-v1/stage1.tlab",
  reranker: "models/element-ranker-v1/reranker.int8.onnx",
  vocab: "models/element-ranker-v1/vocab.txt",
});
const top50 = await ranker.rank("Find a store in Los Angeles", collectCandidates(document));
```

- Run the INT8 re-ranker on ONNX Runtime Web's `wasm` provider; its WebGPU backend scores
  INT8 graphs wrongly in version 1.30.
- WebAssembly threads need a cross-origin isolated page. On one thread a step takes about 1 s.
- Stage 1 and the tokenizer are dependency-free JavaScript (`web/src/`); stage 1 also has a
  C++ runtime (`cpp/`).

## How it was built

1. **Data.** Mind2Web (CC BY 4.0): 7,775 recorded steps on 73 websites for training and
   development, split by website; the three test splits held back.
2. **Stage 1.** A 6,144 x 320 table of hashed byte n-grams, trained with a softmax over each
   page's candidates. Fast and wide, but it memorizes training websites.
3. **Stage 2.** BERT-tiny (4.4M params, Apache-2.0) fine-tuned as a cross-encoder on stage
   1's hard negatives. Pretraining is what lets it generalize to new websites.
4. **Export.** INT8 for both stages; a single-file `.tlab` format for stage 1 and ONNX for
   stage 2. Browser, C++ and Python agree (`docs/benchmarks/browser-runtime.md`).
5. **Choice of size.** BERT-mini was 6 points better on dev but took 1.2 s per step in the
   browser; BERT-tiny fits a one-second budget with room.

## Before the element ranker

tinylab started as a 2 MB tokenizer-free text embedder for hybrid search. It adds +0.024
nDCG@10 to BM25 on technical prose but falls below BM25 on design documents, and no variant
tried (gradient training, bigger tables, multiple hashes, word-order models) closed the gap.
Its weights became stage 1's starting point. The record is in
[docs/benchmarks/search-student.md](docs/benchmarks/search-student.md).

## Repository

```
tinylab/
├── models/element-ranker-v1/   the release: weights, manifest, model card, licence
├── web/                        browser runtime: src/ (tinylab.js, wordpiece.js, element-ranker.js), tests, bench pages
├── cpp/                        C++ runtime for the .tlab format, parity tests
├── python/tinylab/
│   ├── agent/                  element ranker: Mind2Web data, training, re-ranker, export, evaluation
│   ├── train/                  the text student: closed-form fit, distillation
│   ├── export/                 .tlab writer (INT8, budget-checked)
│   ├── eval/                   retrieval evaluation harness, stdlib only
│   ├── data/                   corpora, teacher labelling, word features
│   └── encoder.py              byte n-gram hashing, the frozen input contract
├── tests/                      golden vectors shared by Python, JavaScript and C++
└── docs/benchmarks/            measurements behind every number above
```

## Reproduce

```sh
uv sync --extra train --extra teacher --extra export
uv run tinylab-build-mind2web                   # Mind2Web train split, about 6 GB downloaded
uv run tinylab-train-ranker --tag v1            # stage 1, about 20 min on an M4 Pro
uv run tinylab-train-reranker --tag v1          # stage 2, about 6 min
uv run tinylab-export --checkpoint checkpoints/ranker-linear-6144x320-v1 --out dist/tinylab-ranker-v1
uv run tinylab-export-reranker --reranker checkpoints/reranker-bert_uncased_L-2_H-128_A-2-v1
uv run tinylab-rank-elements --checkpoint checkpoints/ranker-linear-6144x320-v1-int8 \
    --reranker dist/reranker-bert_uncased_L-2_H-128_A-2-v1/reranker.int8.onnx

node web/test/parity.mjs && node web/test/wordpiece-parity.mjs
cmake -S cpp -B cpp/build && cmake --build cpp/build && ctest --test-dir cpp/build
```

Requires [uv](https://docs.astral.sh/uv/), Node.js, a C++20 compiler and CMake. Training
uses MLX and PyTorch on Apple Silicon. Writing code here follows [AGENTS.md](AGENTS.md).

## Licence

Code: MIT ([LICENSE](LICENSE)). Model weights: Apache-2.0
([models/element-ranker-v1/LICENSE](models/element-ranker-v1/LICENSE)), with credits for
Google's BERT miniatures, Mind2Web and the training text in
[NOTICE](models/element-ranker-v1/NOTICE).
