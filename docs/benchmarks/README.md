# Benchmarks

Every number behind the release, with how it was measured and how to reproduce it.

| Page | What it covers | Headline |
| :--- | :--- | :--- |
| [spotlit.md](spotlit.md) | The released model on Mind2Web: data, training, every variant, dev and test | 80.8 / 75.8 / 75.9% of clicked elements in the top 50 on the three test splits; BM25 about 44% |
| [browser-runtime.md](browser-runtime.md) | Export accuracy, cross-language parity, speed in Chrome and C++ | 0.24 s per page step in Chrome (8 threads); INT8 changes r@50 by under 0.01 |
| [demo.md](demo.md) | The interactive browser demo | 36/36 steps agree with Python; 29/36 kept in the top 50 |
| [search-student.md](search-student.md) | The first goal: a 2 MB tokenizer-free embedder for hybrid search | +0.024 nDCG@10 over BM25 on the Rust book; below BM25 on design documents |

## Ground rules

- Development decisions use dev splits; test splits are opened once, after the files they
  measure are frozen.
- Comparisons between systems are paired: same queries, same candidates.
- Results that did not work are recorded with their numbers.
