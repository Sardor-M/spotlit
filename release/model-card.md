---
license: apache-2.0
language:
- en
base_model: google/bert_uncased_L-2_H-128_A-2
datasets:
- osunlp/Mind2Web
pipeline_tag: text-ranking
tags:
- web-agents
- browser-agents
- reranker
- cross-encoder
- onnx
- mind2web
---

# spotlit v1

spotlit ranks the elements of a web page against a browser agent's goal, in the browser, so
the agent's language model reads the 50 elements that matter instead of the whole page. It has
two stages, ships as 6.7 MB, and needs no server and no GPU: about a quarter of a second per
page on eight WebAssembly threads.

On Mind2Web's three test splits, run once after the model was frozen, the element a person
actually used (clicked, typed into or chose from) is in the top 50 on 80.8%, 75.8% and 75.9%
of steps (cross-task, cross-website, cross-domain), against about 44% for keyword search, while
the agent reads 32 to 43 times fewer tokens.

Source, training code and the browser demo: {{GITHUB}}

## Files

| Path | Size | What it is |
| :--- | ---: | :--- |
| `web/spotlit-v1.weights.gz` | 5.04 MB | Everything a web page needs in one gzipped file, read by `spotlit.loadPacked`. spotlit's own format, not a PyTorch checkpoint. |
| `web/spotlit.js`, `tinylab.js`, `wordpiece.js` | | The browser runtime (MIT) |
| `onnx/reranker.int8.onnx` | 4.49 MB | Stage 2 as shipped: BERT-tiny cross-encoder, INT8 ONNX |
| `reranker/` | 17.5 MB | Stage 2 at full precision: `model.safetensors`, `config.json` and tokenizer, for `transformers` |
| `stage1/stage1.tlab` | 1.99 MB | Stage 1 as shipped: hashed byte n-gram table, INT8 |
| `stage1/model.safetensors` | 1.99 MB | The same table and scales as safetensors, for inspection |
| `stage1/config.json` | | The encoder that turns text into table rows |
| `config.json` | | The release at a glance: where each stage is, and the operating point (stage 2 re-reads 400, returns 50) |
| `manifest.json` | | sha256 of every file, input formats, operating point |

## How it works

1. **Describe.** Each element becomes one line: its tag, then its labels (aria-label, title,
   alt, placeholder, name, type, role, value; 60 characters each), then up to 24 words of its
   own text, then `|` and up to 12 words of its parent's text.
2. **Stage 1** embeds the goal and every element with the hashed n-gram table and ranks them by
   cosine similarity. About 35 ms for 500 elements.
3. **Stage 2** reads `[CLS] goal [SEP] element [SEP]` (128 tokens) for stage 1's top 400 and
   re-scores them. About 185 ms with eight WebAssembly threads.
4. The agent receives the top 50.

The goal is the task plus up to three previous actions: `goal + " Previous actions: " + a; b; c`.

## Use

### In a web page

```js
import * as ort from "onnxruntime-web";   // 1.30.0, WebAssembly backend
import * as spotlit from "./spotlit.js";  // from web/ in this repository

const ranker = await spotlit.loadPacked(ort, "spotlit-v1.weights.gz");
const top50 = await ranker.rank(goal, spotlit.collectCandidates(document), {
  k: 50, history: lastThreeActions,
});
```

`loadPacked` checks every part against its sha256. Keep the INT8 re-ranker on the
WebAssembly backend: in ONNX Runtime Web 1.30 it scores wrongly on WebGPU. Threads need a
cross-origin isolated page (COOP and COEP headers); without them a page takes about a second.

### Stage 2 in Python

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo = "{{REPO}}"
tokenizer = AutoTokenizer.from_pretrained(repo, subfolder="reranker")
model = AutoModelForSequenceClassification.from_pretrained(repo, subfolder="reranker").eval()

goal = "Add the cheapest tent to the cart"
elements = [
    "button Add to cart | Trail 1-person tent $129.00 Add to cart",
    "a Returns policy | Shipping Returns policy Contact us",
]
batch = tokenizer([goal] * len(elements), elements, padding=True, truncation=True,
                  max_length=128, return_tensors="pt")
with torch.no_grad():
    scores = model(**batch).logits[:, 0]   # higher is more relevant
```

Element lines must follow the format above. Stage 2 alone re-scores whatever candidates it is
given; the released system re-scores stage 1's top 400.

### Stage 1 outside the browser

The table is only useful with its encoder, which turns text into rows. `stage1/config.json`
specifies it: byte 3, 4 and 5-grams of normalized text, hashed with 64-bit FNV-1a into 6,144
signed buckets. `web/tinylab.js` implements it in about 100 lines of dependency-free
JavaScript, and the GitHub repository has the Python and C++ versions.

## Evaluation

Mind2Web, share of steps whose target element (the one the person clicked, typed into or chose
from) is in the top k. The test splits were run once,
after the files were frozen.

| Split | Steps | Keyword search (BM25) r@50 | Stage 1 r@50 | r@10 | **r@50** | r@100 | Fewer tokens, top 50 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dev (held-out training websites) | 1,106 | 38.2% | 55.2% | 49.3% | **73.1%** | 80.1% | 46x |
| Test, cross-task | 2,094 | 44.2% | 69.2% | 57.2% | **80.8%** | 85.8% | 43x |
| Test, cross-website | 1,373 | 43.8% | 56.7% | 48.7% | **75.8%** | 82.3% | 34x |
| Test, cross-domain | 5,911 | 44.5% | 57.7% | 51.5% | **75.9%** | 83.6% | 32x |

For reference, MindAct's DeBERTa-v3-base ranker (the benchmark authors', 184M parameters with
its embedding table) reports 88.9, 85.3 and 85.7% r@50 on the test splits. Candidate
descriptions differ, and steps whose target element is missing from the page HTML (4 to 7%)
count as misses here, so the comparison is approximate.

Inside Claude Code (Sonnet), on seven multi-page browser tasks, an agent reading spotlit's top
50 used 427,787 input tokens against 2,700,384 when reading every element, and both versions
finished all seven tasks.

## Limits

- About one step in four loses the target element from the top 50. An agent should be able to
  ask for more elements or fall back to the full page.
- Trained on English websites from 2023; other languages and very different page structures
  are untested.
- The reference element collector does not look inside iframes or shadow DOM.
- Token counts are estimates at four characters per token.

## Training

Mind2Web's train split only: 6,669 steps for training and 1,106 steps from held-out websites
for development. The test splits were used for evaluation only. Stage 1 started from a tinylab text-search table distilled
from `BAAI/bge-small-en-v1.5` and was trained with a softmax over each page's candidates for
three epochs. Stage 2 is `google/bert_uncased_L-2_H-128_A-2` fine-tuned as a cross-encoder on
the target element and 31 others from stage 1's top 400, for three epochs. INT8 export changed
development r@50 by +0.6 points.

## License

The model files are Apache-2.0 (`LICENSE`, `NOTICE`). The runtime in `web/` is MIT
(`web/LICENSE`). spotlit was fine-tuned from Google's BERT-tiny (Apache-2.0) and trained on
Mind2Web (CC BY 4.0); `NOTICE` lists every source.

## Citation

```bibtex
@misc{spotlit2026,
  title  = {spotlit: an in-browser element ranker for web agents},
  author = {Madaminov, Sardor},
  year   = {2026},
  month  = {October},
  url    = {{{GITHUB}}}
}
```

## References

1. Deng et al. [Mind2Web: Towards a Generalist Agent for the Web](https://arxiv.org/abs/2306.06070). NeurIPS 2023.
2. Weinberger et al. [Feature Hashing for Large Scale Multitask Learning](https://arxiv.org/abs/0902.2206). ICML 2009.
3. Turc et al. [Well-Read Students Learn Better: On the Importance of Pre-training Compact Models](https://arxiv.org/abs/1908.08962). 2019.
