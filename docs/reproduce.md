# Reproduce spotlit

From the Mind2Web download to the released files. Commands run from the repository root.
Writing code here follows [AGENTS.md](../AGENTS.md).

## Setup

Python 3.12 or 3.13 with [uv](https://docs.astral.sh/uv/), Node.js (tested on 23), and for the
C++ runtime a C++20 compiler and CMake. Training runs on Apple Silicon: stage 1 on MLX,
stage 2 on PyTorch with MPS.

```sh
uv sync --extra train --extra teacher --extra export
```

The `train` extra brings MLX, `teacher` brings Sentence Transformers (for the teacher model
and BERT-tiny), and `export` brings ONNX and ONNX Runtime.

## Data

```sh
uv run tinylab-build-mind2web                                # train split, about 6 GB downloaded
uv run tinylab-build-mind2web --test-password <password>     # also the three test splits
```

The data is [Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web) (CC BY 4.0): 7,775
recorded steps on 73 websites in its training split, and three test splits. Nothing is
redistributed here; the builder downloads the source files, describes every candidate
element, and writes one line per step:

```
data/mind2web/train.jsonl        6,669 steps   training websites
data/mind2web/dev.jsonl          1,106 steps   10 websites never trained on, chosen by a salted hash
data/mind2web/test_task.jsonl    2,094 steps   } built only with --test-password,
data/mind2web/test_website.jsonl 1,373 steps   } used for evaluation only,
data/mind2web/test_domain.jsonl  5,911 steps   } and opened once
```

Each line holds the task, the previous actions, the operation, the candidate elements as
`[id, description]`, the clicked element's id and the size of the step's cleaned HTML. The
test password is in the Mind2Web README; its authors ask that the unzipped test data is never
redistributed or used for training.

Stage 1 starts from an earlier model, a 2 MB text embedder fitted on public technical and
design documents (six CQADupStack forums; Rust RFCs, Kubernetes enhancement proposals and
docs, Go and Swift proposals) labelled by `BAAI/bge-small-en-v1.5`. The commands to rebuild it are in
[python/tinylab/train](../python/tinylab/train/README.md); its results are in
[docs/benchmarks/search-student.md](benchmarks/search-student.md).

## Training

```sh
uv run tinylab-fit --tag design                  # the starting table, about a minute after labelling
uv run tinylab-train-ranker --tag v1             # stage 1, about 20 minutes on an M4 Pro
uv run tinylab-train-reranker --tag v1           # stage 2, about 6 minutes
```

**Stage 1: the n-gram table.** Starting from the 6,144 x 320 embedder table, it trains on
one page per update with a softmax over the clicked element and every other candidate on the
page, at temperature 0.05. Adam, learning rate 5e-4 with cosine decay, 3 epochs (18,804
updates), peak MLX memory 0.34 GB. Every 1,000 updates it logs recall on 400 dev steps.

**Stage 2: the re-ranker.** `google/bert_uncased_L-2_H-128_A-2` (BERT-tiny, Apache-2.0) as a
cross-encoder with 128-token pairs. Each batch holds 8 steps of 32 pairs: the clicked element
and 31 negatives drawn from stage 1's own top 400, with a softmax per step. AdamW, learning
rate 1e-4 with linear warmup and decay, 3 epochs (2,350 batches), 1.1 GB on MPS.
`--base google/bert_uncased_L-4_H-256_A-4` trains the BERT-mini variant.

## Export

```sh
uv run tinylab-export --checkpoint checkpoints/ranker-linear-6144x320-v1 --out dist/tinylab-ranker-v1
uv run tinylab-export-reranker --reranker checkpoints/reranker-bert_uncased_L-2_H-128_A-2-v1
```

The first writes stage 1 as a single-file `.tlab` (INT8 rows with one scale each, format in
[python/tinylab/export](../python/tinylab/export/README.md)). The second writes the re-ranker as
FP32 and dynamically quantized INT8 ONNX, with `vocab.txt` and goldens (token ids and scores)
that the JavaScript runtime is checked against.

## Evaluation

```sh
uv run tinylab-rank-elements --teacher                                   # BM25 and untrained baselines on dev
uv run tinylab-rank-elements --checkpoint checkpoints/ranker-linear-6144x320-v1-int8 \
    --reranker dist/reranker-bert_uncased_L-2_H-128_A-2-v1/reranker.int8.onnx
uv run tinylab-rank-elements --steps data/mind2web/test_website.jsonl \
    --checkpoint checkpoints/ranker-linear-6144x320-v1-int8 \
    --reranker dist/reranker-bert_uncased_L-2_H-128_A-2-v1/reranker.int8.onnx
```

`tinylab-rank-elements` prints recall@1, 5, 10, 20, 50, 100 and 200, MRR and the token ratio
for BM25 and for each `--checkpoint`; `--reranker` re-ranks the first checkpoint's top 400.
Ties are broken by element id in stage 1 and by stage 1 rank in stage 2, never by the order of
Mind2Web's candidate list, which puts the clicked element first.

```sh
node web/test/parity.mjs                                                  # encoder and .tlab vs Python
node web/test/wordpiece-parity.mjs dist/reranker-bert_uncased_L-2_H-128_A-2-v1
cmake -S cpp -B cpp/build && cmake --build cpp/build && ctest --test-dir cpp/build
```

`web/bench-rerank.html` times a page step in the browser. Serve the repository with
`Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`
headers to get threads.
