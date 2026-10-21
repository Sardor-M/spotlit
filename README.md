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

## Benchmarks

The numbers come from [Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web): people completing
tasks on real websites, one clicked element per step. Recall@k is the share of steps whose
clicked element is in the top k. The three test splits were opened once, after the files were
frozen; every decision was made on a separate dev split.

**Recall@50 on the test splits**

| System | Size | Cross-task | Cross-website | Cross-domain |
| :--- | ---: | ---: | ---: | ---: |
| Keyword search (BM25) | 0 | 44.2% | 43.8% | 44.5% |
| Stage 1 alone | 2.0 MB | 69.2% | 56.7% | 57.7% |
| **spotlit v1** | **6.7 MB** | **80.8%** | **75.8%** | **75.9%** |
| MindAct DeBERTa-v3-base, as reported in the Mind2Web paper | 184M params | 88.9% | 85.3% | 85.7% |

**spotlit per test split**

| Split | Steps | Recall@10 | Recall@50 | Recall@100 | MRR | Tokens vs the cleaned HTML |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cross-task | 2,094 | 57.2% | **80.8%** | 85.8% | 0.274 | 43x fewer |
| Cross-website | 1,373 | 48.7% | **75.8%** | 82.3% | 0.219 | 34x fewer |
| Cross-domain | 5,911 | 51.5% | **75.9%** | 83.6% | 0.251 | 32x fewer |

**In the browser and inside an agent**

| Measure | Every element | spotlit |
| :--- | ---: | ---: |
| Page step in Chrome, 8 WebAssembly threads, Apple M4 Pro | | **236 ms** |
| Page step on one thread (no cross-origin isolation) | | about 1 s |
| Claude Code (Claude Sonnet, one run per task), seven multi-page tasks: input tokens | 2,700,384 | **427,787** |
| Cost and time for the seven tasks | $3.55, 149 s | **$0.48**, 173 s |
| Tasks finished | 7 of 7 | 7 of 7 |

Cross-website and cross-domain are the numbers to trust: websites and whole domains the model
never trained on, where spotlit keeps the clicked element 31-32 points more often than keyword
search. Cross-task is higher because its websites appear in training. MindAct's ranker is 29
times larger and keeps 8-10 points more.

Full results: [docs/benchmarks/spotlit.md](docs/benchmarks/spotlit.md) (dev results, every
variant tried, why BERT-tiny), [docs/benchmarks/browser-runtime.md](docs/benchmarks/browser-runtime.md)
(speed and parity across Python, JavaScript and C++) and [browser-agent](browser-agent/README.md)
(the agent runs, task by task).

## Strengths and weaknesses

spotlit makes a shortlist: it is small and fast, and keeps the right element in the top 50 on
most steps of websites it never saw. It does not pick the action itself, and it misses about
one step in four on new websites, so an agent should keep a fallback.

| Strength | Evidence |
| :--- | :--- |
| **Small enough to ship with a page** | 6.7 MB in three files. INT8 export moved dev recall@50 by +0.6 points |
| **No server and no GPU** | WebAssembly in Chrome: 236 ms per page step on 8 threads, about 1 s on one |
| **Cuts what the agent reads** | The top 50 is 32-43x fewer tokens than the cleaned HTML on the test splits |
| **Holds up on websites it never saw** | Cross-website and cross-domain recall@50 of 75.8% and 75.9%, 31-32 points above BM25 |
| **Tolerant of spelling and formatting** | Stage 1 reads bytes: "Tokio tee ceremony" and "Tokyo tea ceremony" have cosine 0.69 |

| Weakness | Evidence | What it means |
| :--- | :--- | :--- |
| **A shortlist, not a picker** | Dev recall@1 is 10.9% and recall@10 is 49.3% | The agent's model still has to choose among the 50 |
| **Misses about one step in four on new websites** | Cross-website recall@50 is 75.8%; recall@100 is 82.3% | An agent needs a fallback: a larger k, or the full page |
| **Behind larger rankers** | MindAct's 184M-parameter ranker reports 8-10 points more recall@50 | BERT-mini (13.4 MB) was 6 points better on dev but took 1.2 s per step |
| **Stage 1 caps stage 2** | Stage 1's top 400 holds the clicked element on 87.6% of dev steps | The other 12.4% can't be recovered by the re-ranker |
| **Look-alikes** | For "Sign in to my account", stage 2 ranks "Sign up free" first and "Log in" fifth | Unusual wording on a site can mislead it |

It was trained on English websites from 2023 and reads text only, with no screenshot, position
or visibility. Elements inside iframes and shadow roots are not collected.

Every strength, weakness and limitation, with its evidence:
[docs/strengths-and-limits.md](docs/strengths-and-limits.md).

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

## Train it yourself

This repository holds the code that builds the training data from Mind2Web, trains both
stages, exports them to INT8, evaluates them, and runs them in JavaScript and C++. The
commands, named `tinylab-*` after the training toolkit in `python/tinylab`, are in
[docs/reproduce.md](docs/reproduce.md).

## Licence

Code: MIT ([LICENSE](LICENSE)). Model weights: Apache-2.0
([models/spotlit-v1/LICENSE](models/spotlit-v1/LICENSE)), with credits for Google's BERT
miniatures, Mind2Web and the training text in [NOTICE](models/spotlit-v1/NOTICE).
