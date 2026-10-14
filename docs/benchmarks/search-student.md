# The search student: a 2 MB tokenizer-free text embedder

tinylab's first goal was a text embedding model under 2 MB, with no tokenizer, that adds to
keyword search (BM25) in hybrid retrieval. This page records what it reached and what did
not work. Its weights became the starting point of the element ranker's stage 1.

## Setup

- **Teacher.** `BAAI/bge-small-en-v1.5` (33M params, MIT).
- **Student.** Byte 3/4/5-grams hashed into a 6,144 x 320 table, mean-pooled, L2-normalized;
  fitted in closed form (ridge regression onto the teacher's top 320 principal components)
  on about 550K labelled texts: CQADupStack forums and design documents (RFCs, KEPs, Go and
  Swift proposals, Kubernetes docs). 1.99 MB at INT8.
- **Evaluation.** Hybrid = reciprocal rank fusion of BM25 and the student (k = 60), nDCG@10,
  paired bootstrap. The bar: hybrid beats BM25 by at least +0.03 with P >= 0.95.
- **Corpora.** A private corpus built from the author's project documentation (section
  headings as queries, 943 queries over 1,271 sections; not released), and the public *Rust
  Programming Language* book (458 heading queries over 511 sections, MIT / Apache-2.0).

## Results

| Corpus | BM25 | Student | BM25 + student | Teacher | BM25 + teacher |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Rust book (458 queries) | 0.589 | 0.544 | **0.613** | 0.698 | 0.687 |
| Private docs, dev (597 queries) | 0.265 | 0.165 | 0.262 | 0.261 | 0.321 |
| Private docs, val (128 queries) | 0.219 | 0.119 | 0.192 | 0.230 | 0.264 |

On technical prose like its training text, the student adds +0.024 to BM25. On dense design
documents it lowers the score, while the teacher still adds +0.045 (P 0.992). The bar was
not met.

## What was tried

Effect on hybrid nDCG@10 on dev. Rows marked v1 were measured on the first, smaller version of
the private corpus (222 dev queries), the rest on the current one (597).

| Idea | Set | Effect |
| :--- | :--- | :--- |
| Gradient-trained students (MSE, listwise, lexical-target mixes) | v1 | all below BM25 |
| Adding web text (MS MARCO) to technical text | v1 | -0.028 and -0.022 at the two table sizes tried |
| More CQADupStack forums | v1 | stats: 0; tex and mathematica: -0.009 |
| Embedding whole sections as 512-byte windows | v1 | -0.004 to -0.017 |
| Two or three hash functions per n-gram | v1 | -0.005 / -0.016 |
| Design-document training text | v2 | +0.007 |
| Twice the table (12,288 buckets, the capacity INT4 would buy) | v2 | +0.005 |
| Unlimited 100K-word static vocabulary from the teacher | v2 | -0.025 vs the student |
| Contextual student, 1.9M / 7.5M params (words in order, transformer) | v2 | -0.016 / +0.002 vs the student |

Another hash function alone moved the score by 0.015 (v1), so differences below that are
noise. The conclusion: without pretraining, a model this small learns too little language
from half a million labelled texts to beat BM25 on varied documentation.

## Speed and size

1,990,696 bytes INT8. In Chrome a query embeds in 0.019 ms and a passage in 0.165 ms; the
JavaScript and C++ runtimes match Python to 5e-9 and 2.7e-8 (`browser-runtime.md`).
