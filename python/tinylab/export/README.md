# tinylab.export

Turns the float linear student into the one file the runtimes load (ADR-0009).

```sh
uv run tinylab-export                      # checkpoints/student-linear-6144x320-design
```

Writes `dist/tinylab-student-v1/`:

| File | Contents |
| :--- | :--- |
| `student.tlab` | the model: header, per-row scales, INT8 table (1,990,696 bytes) |
| `model.json` | encoder spec, size, sha256, quantization error |
| `goldens.json` | reference embeddings for runtime parity tests |

It also writes `checkpoints/<stem>-int8`, the dequantized weights, so `tinylab-eval` scores
exactly what ships. Export refuses a file over 2,000,000 bytes (ADR-0008).

## `.tlab` version 1

Little-endian. Scales start 4-byte aligned, so a browser can view them in place.

| Offset | Field |
| :--- | :--- |
| 0 | magic `TLAB` |
| 4 | u32 format version (1) |
| 8 | u32 encoder version (1) |
| 12 | u32 bucket count B |
| 16 | u32 output dimensions D |
| 20 | u32 max input bytes |
| 24 | u32 n-gram size count N, then N u32 sizes |
| 28 + 4N | f32 scales[B] |
| 28 + 4N + 4B | i8 table[B][D] |

## Embedding a text

1. Hash the text into signed bucket counts with the v1 encoder (`tinylab.encoder`, ADR-0005).
2. Sum `count * scale[b] * table[b]` over the buckets present.
3. L2-normalize. Cosine between two embeddings is their dot product.
