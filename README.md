# tinylab

Building very small models that run on device — in a browser tab, in a C++ binary, on a
microcontroller — and releasing them as open weights.

In development. The harness, encoder and teacher measurements exist; the student is a
closed-form linear fit that beats BM25 in fusion on development queries but has not yet
passed the evaluation gate. Nothing is released.

**Current target:** a tokenizer-free text embedding model — short text to a ~320 dim
vector, distilled from a strong sentence embedder, with hashed character n-grams instead of
a vocabulary file. 2 MB INT8, hard budget. First consumer is hybrid retrieval.

## The idea

A large model's judgment over a narrow, fixed-shape input can be compressed into a few
hundred kilobytes, if you distill it densely, quantize it deliberately, and measure it
honestly.

The reference artifact is [millionco/moka](https://github.com/millionco/moka): a
105,353-parameter Go network, 114 KB of INT8 weights, running in an ordinary webpage with
no ML framework on the deployment side. The domain is interchangeable; Go is just the demo.

```
   TEACHER                        STUDENT                     ARTIFACT
   large, slow, accurate          small, fast, close enough   what ships

   +-------------------+          +-----------------+         +----------------+
   |  1M-1B params     |          |  ~100K-2M params|         |  ~0.1-2 MB     |
   |  megabytes        |  ---->   |  fixed shape    |  ---->  |  INT8 weights  |
   |  needs a runtime  | distill  |  no framework   | quantize|  + manifest    |
   +-------------------+          +-----------------+         +----------------+
                                                                     |
                                                                     v
                                               +---------------------+-------------------+
                                               |                     |                   |
                                          C++ binary            browser Worker      microcontroller
                                          (hand-written         (hand-written       (integer kernels,
                                           kernels)              kernels)            no FPU)
```

## Layout

```
tinylab/
├── AGENTS.md           rules for writing code here
├── python/tinylab/     training side
│   ├── encoder.py          text -> hashed byte n-grams, frozen contract
│   ├── eval/               evaluation harness, baselines, keep/discard gate
│   ├── data/               dataset generation, teacher labelling
│   ├── models/             network definitions
│   ├── train/              training loop, losses, quantization-aware training
│   └── export/             quantization, manifest + binary writer
├── cpp/                inference side, hand-written kernels, no dependencies
│   ├── include/tinylab/
│   ├── src/
│   └── tests/              cross-language parity against tests/encoder_goldens.txt
├── web/                browser runtime (Worker + hand-written kernels)
├── models/             exported artifacts and manifests
├── tests/              golden vectors shared by every language port
└── docs/
    └── all/            private notes, gitignored, local only
```

`docs/all/` is deliberately excluded from git. It holds working notes, source teardowns,
decision records and the experiment log. Those are for building understanding, not for
publishing.

## Principles

Carried over from studying the reference implementation.

**Build the evaluation harness before the model.** Everything downstream is only meaningful
if the number is trustworthy. Split held-out data by the natural correlated unit — document,
session, recording — never by row. Seal a final test set and open it once.

**Never select on a proxy metric.** In the reference project, the checkpoint with the best
held-out accuracy was one of its weakest players in actual head-to-head games. Score the
thing you care about, end to end.

**Spend effort at inference time, not only on weights.** Training work moved the reference
model from 2 to 19 arena wins. Search and test-time ensembling moved it from 19 to 51.

**Size budgets are gates, not preferences.** Export fails when the artifact exceeds the
budget, the same way a failing test fails a build.

**Record what did not work, with the number.** A rejected idea you can point at is worth
more than a vague memory that it "didn't work."

## Roadmap

Each phase ends with something measurable.

```
  [0] understand the reference      done
       |
  [1] choose the problem            done — tokenizer-free text embedder
       |
  [2] build the evaluation harness  done — python/tinylab/eval/, zero dependencies
       |
  [3] trivial baseline              done on the fixture; pending a real corpus
       |
  [4] input encoding                done — Python + C++, parity tested
       |
  [5] teacher                       done — premise confirmed, dense beats BM25
       |
  [6] student v0                    any learned model that beats the baseline
       |
  [7] distill properly              dense targets, on-policy correction
       |
  [8] quantize                      INT8 export against a hard byte budget
       |
  [9] serve                         C++ runtime, then browser; latency measured
       |
 [10] release                       open weights, licence, model card, embed example
```

Phase 2 before phase 5 is not a typo. If the harness comes after the first results, the
metric gets chosen to fit numbers you have already seen.

## Getting started

The evaluation harness runs today. The model sides are still stubs that prove the
toolchain.

```sh
# evaluation harness — stdlib only, no install needed
PYTHONPATH=python python3 -m tinylab.eval.cli selfcheck
PYTHONPATH=python python3 -m tinylab.eval.cli run --system bm25 --split val

# training side (uv provisions its own interpreter)
uv sync

# inference side, including the encoder parity test
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure
./cpp/build/tinylab_cli
```

Requires a C++20 compiler, [CMake](https://cmake.org/), and [uv](https://docs.astral.sh/uv/).

Without CMake installed, the scaffold builds directly:

```sh
clang++ -std=c++20 -O3 -Icpp/include cpp/src/*.cpp -o /tmp/tinylab_cli && /tmp/tinylab_cli
```

## Conventions

[AGENTS.md](AGENTS.md) holds the rules for writing code here. The short version: comments
are one line or absent, and reasoning lives in `docs/`, not scattered through source files.

## Licence

MIT for code. The weights licence will be stated separately once there are weights — code
licences do not clearly cover model weights, and leaving that ambiguous is the most common
failure in "open weights" releases.
