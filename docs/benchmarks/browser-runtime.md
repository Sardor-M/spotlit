# Browser and C++ runtimes

Whether the exported files behave like the models they came from, and how fast they run.

## Export accuracy

| Check | Result |
| :--- | :--- |
| Stage 1 `.tlab` INT8 vs float, search task (Lumen dev, fused nDCG@10) | 0.2618 vs 0.2616, 95% CI of the difference [-0.0004, +0.0009] |
| Stage 1 INT8 vs float, element ranking (Mind2Web dev r@50) | 55.2% vs 55.3% |
| BERT-tiny INT8 ONNX vs PyTorch (Mind2Web dev r@50) | 73.1% vs 72.5% |
| BERT-mini INT8 ONNX vs PyTorch (Mind2Web dev r@50) | 78.3% vs 78.7% |

## Parity across languages

| Component | Test | Result |
| :--- | :--- | :--- |
| Byte n-gram encoder, JavaScript | 31 golden cases (`tests/encoder_goldens.txt`) | 31/31 identical |
| Byte n-gram encoder, C++ | same goldens | 31/31 identical |
| `.tlab` embeddings, JavaScript | 34 texts vs the Python reference | max difference 5e-9 |
| `.tlab` embeddings, C++ | same texts | max difference 2.7e-8 |
| WordPiece tokenizer, JavaScript | 667 pairs vs Hugging Face's fast tokenizer: real pairs, edge-case strings (accents, CJK, emoji, control characters), a length grid; 177 truncated | 667/667 identical token ids |
| Whole pipeline, Chrome | 40 dev steps, top-50 membership vs Python | 40/40 agree |
| Whole pipeline, Chrome, demo page | 36 steps of 8 tasks | 36/36 agree |

The pair truncation rule of the Rust tokenizer is not the textbook longest-first rule; the
JavaScript version was fitted to 1,849 length pairs and matches all of them.

## Speed

Chrome 155 (headless), Apple M4 Pro, ONNX Runtime Web 1.30.0, a cross-origin isolated page
with 8 WebAssembly threads; medians over 38 dev steps, 400 pairs re-scored per step.

| Re-ranker | Weights | Backend | Download | Re-score 400 | Whole step | Correct |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| **BERT-tiny** | **INT8** | **WebAssembly** | **4.49 MB** | **185 ms** | **236 ms** | yes |
| BERT-tiny | FP32 | WebGPU | 17.6 MB | 67 ms | 111 ms | yes |
| BERT-tiny | INT8 | WebGPU | 4.49 MB | 288 ms | 322 ms | **no** |
| BERT-mini | INT8 | WebAssembly | 11.37 MB | 1,158 ms | 1,203 ms | yes |
| BERT-mini | FP32 | WebGPU | 44.77 MB | 456 ms | 505 ms | yes |
| BERT-mini | INT8 | WebGPU | 11.37 MB | 1,493 ms | 1,548 ms | **no** |

- Stage 1 takes 35-45 ms of each step and WordPiece tokenization about 10 ms.
- Dynamically quantized INT8 graphs return wrong scores on the WebGPU backend of this ORT
  version (top-50 agreement with Python 9-13 of 40). Use the `wasm` provider for INT8.
- Without cross-origin isolation (one thread, as on most hosted pages) a BERT-tiny step
  takes about 1 s.
- ONNX Runtime Web's engine is a further 14.2 MB download, shared and cacheable.

**The search student alone** (`.tlab`, no ONNX): a query embeds in 0.019 ms and a 512-byte
passage in 0.165 ms in Chrome; 0.009 ms and 0.025 ms in C++ (-O3). The 1.99 MB file loads
in Chrome in 19 ms.

## Reproduce

```sh
node web/test/parity.mjs                                       # encoder and .tlab parity
node web/test/wordpiece-parity.mjs dist/reranker-bert_uncased_L-2_H-128_A-2-v1
cmake -S cpp -B cpp/build && cmake --build cpp/build && ctest --test-dir cpp/build
```

`web/bench-rerank.html` times the pipeline per page step; serve the repository with
`Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`
headers and ONNX Runtime Web at `?ort=` (default `/ortweb/`).
