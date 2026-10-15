# Spotlit on Mind2Web

What the released model (`models/spotlit-v1`) was trained and measured on, and every
variant tried along the way.

## Task

At each step of a recorded web task, a person clicked one element. Given the task and the
previous actions, rank the page's candidate elements so the clicked one comes first. An
agent would send the top k to its large model.

- **Data.** [Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web) (CC BY 4.0). Train split
  only for training. Its websites are split by a salted hash into train (6,669 steps) and
  dev (1,106 steps, 15% of websites), so dev pages come from websites never trained on.
- **Test.** The three official test splits (cross-task 2,094, cross-website 1,373,
  cross-domain 5,911 steps), opened once after the pipeline was frozen.
- **Candidates.** A median of 400-514 elements per step. Each is described from the step's
  cleaned HTML: tag, key attributes, up to 24 words of own text, `|`, 12 words of parent
  text (`python/tinylab/agent/dom.py`).
- **Query.** The task plus up to three previous actions.
- **Metric.** r@k, the share of steps whose clicked element is in the top k; MRR. Steps
  whose clicked element is not among the described candidates (4-7%) count as misses.
- **Tokens.** Characters of the step's cleaned HTML divided by characters of the top-k
  descriptions (median over steps).

## Systems

| System | Size | Trained on Mind2Web |
| :--- | ---: | :--- |
| BM25 over the descriptions | 0 | no |
| bge-small-en-v1.5, cosine | 33M params | no |
| tinylab search student, cosine | 2 MB | no |
| **Stage 1**: tinylab table, softmax over each page's candidates | 2 MB | yes |
| Stage 1 + BERT-tiny cross-encoder on stage 1's top 400 | 6.7 MB | yes |
| Stage 1 + BERT-mini cross-encoder on stage 1's top 400 | 13.4 MB | yes |

**Stage 1 training.** Initialized from the search student (6,144 x 320). One page per
update: softmax over the gold element and all other candidates, temperature 0.05, Adam,
learning rate 5e-4 cosine-decayed, 3 epochs (18,804 updates). 20 minutes on an M4 Pro,
peak MLX memory 0.34 GB.

**Re-ranker training.** `google/bert_uncased_L-2_H-128_A-2` (4.4M params) as a
cross-encoder, 128 tokens. Each batch holds 8 steps of 32 pairs: the gold element and 31
negatives drawn from stage 1's top 400; softmax per step. AdamW, learning rate 1e-4,
linear warmup and decay, 3 epochs (2,350 batches). 5.5 minutes on MPS at 1.1 GB. BERT-mini
(`L-4_H-256_A-4`, 11.2M params) with the same recipe: 16.5 minutes at 1.2 GB.

## Dev results

| System | r@1 | r@10 | r@50 | r@100 | r@200 | MRR | Tokens, top 50 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 3.6% | 13.0% | 38.2% | 50.8% | 64.1% | 0.073 | 21x |
| bge-small, untrained | 3.3% | 12.5% | 31.4% | | | 0.066 | 21x |
| search student, untrained | 1.2% | 5.0% | 14.0% | | | 0.029 | 15x |
| Stage 1 | 7.8% | 31.1% | 55.3% | 66.9% | 79.3% | 0.155 | 34x |
| Stage 1 + BM25 (RRF) | 5.1% | 24.1% | 54.3% | 68.7% | 75.4% | 0.114 | 26x |
| Stage 1 + BERT-tiny | 10.8% | 49.7% | 72.5% | 80.4% | 85.6% | 0.230 | 45x |
| Stage 1 + BERT-mini | 16.5% | 54.2% | 78.7% | 84.2% | 86.6% | 0.285 | 44x |
| **Released, INT8 files** | 10.9% | 49.3% | **73.1%** | 80.1% | 85.6% | 0.229 | 46x |

Notes:

- Stage 1's r@50 on its 400-step logging sample levelled off near 53% from update 11,000
  while training loss kept falling: it memorizes training websites. The pretrained re-ranker is what generalizes.
- Fusing stage 1 with BM25 does not help.
- A re-ranker can only promote what stage 1 returns. Stage 1 holds the clicked element in
  its top 300 / 400 for 84.7% / 87.6% of steps, the ceiling for a re-ranker of that depth.
- INT8 export changed dev r@50 by +0.006 (BERT-tiny) and -0.004 (BERT-mini).

## Test results (released files, run once)

| Split | Steps | BM25 r@50 | Stage 1 r@50 | r@10 | **r@50** | r@100 | MRR | Tokens, top 50 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cross-task | 2,094 | 44.2% | 69.2% | 57.2% | **80.8%** | 85.8% | 0.274 | 43x |
| Cross-website | 1,373 | 43.8% | 56.7% | 48.7% | **75.8%** | 82.3% | 0.219 | 34x |
| Cross-domain | 5,911 | 44.5% | 57.7% | 51.5% | **75.9%** | 83.6% | 0.251 | 32x |

Dev agrees with cross-website and cross-domain within 3 points. Cross-task is higher because
its websites appear in training. MindAct's DeBERTa-v3-base candidate ranker (86M backbone
parameters, 184M with its embedding table) reports 88.9 / 85.3 / 85.7% r@50 on these splits (Mind2Web paper, section 4.3); descriptions
and the treatment of undescribed elements differ, so the gap of 8-10 points is approximate.

## Why BERT-tiny and not BERT-mini

BERT-mini is 6 points better on dev, but re-scoring 400 pairs takes 1.16 s in Chrome at INT8
against 0.19 s for BERT-tiny (`browser-runtime.md`). The release targets one second per page
step on WebAssembly, which BERT-tiny meets with room.

## Reproduce

```sh
uv sync --extra train --extra teacher --extra export
uv run tinylab-build-mind2web                       # 6 GB download, processed file by file
uv run tinylab-rank-elements --teacher              # untrained baselines on dev
uv run tinylab-train-ranker --tag v1                # stage 1, about 20 min
uv run tinylab-train-reranker --tag v1              # BERT-tiny, about 6 min
uv run tinylab-export --checkpoint checkpoints/ranker-linear-6144x320-v1 --out dist/tinylab-ranker-v1
uv run tinylab-export-reranker --reranker checkpoints/reranker-bert_uncased_L-2_H-128_A-2-v1
uv run tinylab-rank-elements --checkpoint checkpoints/ranker-linear-6144x320-v1-int8 \
    --reranker dist/reranker-bert_uncased_L-2_H-128_A-2-v1/reranker.int8.onnx
```

The test splits are built with `tinylab-build-mind2web --test-password <password>`; the
password is in the Mind2Web repository's README. Their authors ask that the unzipped test
data is never redistributed or used for training.
