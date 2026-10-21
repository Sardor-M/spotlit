<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/spotlit-mark-dark.png">
  <img alt="spotlit" src="docs/img/spotlit-mark-light.png" height="48">
</picture>

[Live demo](https://spotlit-v1.pages.dev) &middot; [Blog post](https://spotlit-v1.pages.dev/blog/spotlit/) &middot;
[Weights on Hugging Face](https://huggingface.co/Sardori/spotlit-v1) &middot; [Model card](models/spotlit-v1/README.md)

spotlit ranks a web page's elements against an AI browser agent's goal, so the agent's
language model reads the 50 elements that matter instead of the whole page. It is 6.7 MB of
open weights, runs in the browser on WebAssembly with no server and no GPU, and takes about a
quarter of a second per page.

| Stage | Size | What it does |
| :--- | ---: | :--- |
| 1. Hashed byte n-grams | 2.0 MB | Scores every element by cosine with the goal. No tokenizer, so typos and case change little |
| 2. BERT-tiny cross-encoder | 4.7 MB | Reads the goal and each of stage 1's top 400 together, and keeps the best 50 |

## How it works

```
   web page: hundreds to thousands of elements
        |
   describe     each element as one line: tag, labels, own text | parent text
        |
   stage 1      2.0 MB  hashed byte n-grams, cosine with the goal       every element, 35-45 ms
        |  top 400
   stage 2      4.7 MB  BERT-tiny reads goal and element together       400 pairs, 185 ms
        |  top 50
   the agent's large model picks the action: click, type, select
```

## Results

Recall@50 on Mind2Web's held-out test splits, run once after the model was frozen: how often
the element a person actually clicked is in the top 50.

| System | Size | Cross-task | Cross-website | Cross-domain |
| :--- | ---: | ---: | ---: | ---: |
| Keyword search (BM25) | 0 | 44.2% | 43.8% | 44.5% |
| Stage 1 alone | 2.0 MB | 69.2% | 56.7% | 57.7% |
| **spotlit v1** | **6.7 MB** | **80.8%** | **75.8%** | **75.9%** |
| MindAct DeBERTa-v3-base, as reported in the Mind2Web paper | 184M params | 88.9% | 85.3% | 85.7% |

The top 50 is 32-43x fewer tokens than the page's cleaned HTML. A page step takes 236 ms on
eight WebAssembly threads in Chrome on an Apple M4 Pro. In Claude Code, on seven multi-page
tasks, the agent read 427,787 input tokens instead of 2,700,384 and finished all seven either
way. Every measurement is in [docs/benchmarks](docs/benchmarks/README.md).

## Use it

```js
import * as ort from "onnxruntime-web";
import * as spotlit from "./web/src/spotlit.js";

const ranker = await spotlit.loadPacked(ort, "spotlit-v1.weights.gz");
const top50 = await ranker.rank("Find a store in Los Angeles", spotlit.collectCandidates(document));
```

`spotlit-v1.weights.gz` (5.0 MB) holds both stages. Download it from the
[latest release](../../releases/latest) or Hugging Face, or build it with `node web/pack.mjs`.
Serve the page with `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` to get eight threads; without them it runs on
one, about a second per page. The full API is in [web/README.md](web/README.md).

- **Python:** the Hugging Face release has the full-precision re-ranker for `transformers`.
- **In an agent:** [browser-agent/](browser-agent/README.md) runs spotlit inside Claude Code
  or Codex through a small MCP browser server.

## Limitations

- A shortlist, not a picker: the clicked element is first on about 11% of dev steps, so the
  agent's model still chooses among the 50.
- About one step in four on a website it never saw misses the top 50; an agent needs a
  fallback, such as a larger k or the full page.
- Trained on English websites from 2023, as text only: no screenshot, position or visibility.
- Elements inside iframes and shadow roots are not collected.

The full list, with the evidence for each: [docs/strengths-and-limits.md](docs/strengths-and-limits.md).

## Train it yourself

This repository holds the code that builds the training data from Mind2Web, trains both
stages, exports them to INT8, evaluates them, and runs them in JavaScript and C++. The
commands, named `tinylab-*` after the training toolkit in `python/tinylab`, are in
[docs/reproduce.md](docs/reproduce.md).

## Licence

Code: MIT ([LICENSE](LICENSE)). Model weights: Apache-2.0
([models/spotlit-v1/LICENSE](models/spotlit-v1/LICENSE)), with credits for Google's BERT
miniatures, Mind2Web and the training text in [NOTICE](models/spotlit-v1/NOTICE).
