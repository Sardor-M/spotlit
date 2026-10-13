# tinylab.train

Distills the teacher into the student (ADR-0006). The working student is the closed-form
linear fit (`tinylab-fit`); `tinylab-train` is the gradient path. One GPU job at a time: a
second `tinylab-fit`, `tinylab-train` or `tinylab-label` refuses to start while one is running.

## Training text

Labelled jsonl files in `data/train/`, teacher vectors beside each as `<stem>-teacher.npy`.
Repositories for the design set are cloned into `data/sources/` and licence-checked first
(ADR-0007).

```sh
uv run tinylab-build-technical-set --forums stats --name stats
uv run tinylab-build-design-set data/sources/rfcs data/sources/enhancements \
    data/sources/proposal data/sources/swift-evolution data/sources/website
caffeinate -i uv run tinylab-label --passages data/train/design-passages.jsonl \
    --out data/train/design-passages-teacher.npy
caffeinate -i uv run tinylab-label --passages data/train/design-titles.jsonl \
    --out data/train/design-titles-teacher.npy --query-prefix
```

## Fit

```sh
caffeinate -i nice -n 10 uv run tinylab-fit --tag design \
    --source data/train/tech-passages.jsonl --source data/train/tech-titles.jsonl \
    --source data/train/stats-passages.jsonl --source data/train/stats-titles.jsonl \
    --source data/train/design-passages.jsonl --source data/train/design-titles.jsonl
```

About a minute; writes `checkpoints/student-linear-6144x320-<tag>`. Without `--source` it
fits the defaults (tech + stats + design), the release model.

## Contextual student (ADR-0011)

Hashed word pieces read in order by a small transformer, distilled by cosine to the
teacher's vectors on tech + stats + design text. Two probe sizes: `S` (1.9M params, 2 MB at
INT8) and `M` (7.5M). The first run per size builds word caches beside each jsonl.

```sh
caffeinate -i nice -n 10 uv run tinylab-train-contextual --size S --tag probe \
    2>&1 | tee runs/ctx-S-probe.log
```

Writes `checkpoints/student-ctx-S-seed0-probe`; score it like any student checkpoint.

## Train (gradient)

From the repository root, plugged in, nothing else heavy running:

```sh
mkdir -p runs
caffeinate -i nice -n 10 uv run tinylab-train --config A --seed 0 --steps 16000 \
    --queries data/train/queries.jsonl --tag q50 2>&1 | tee runs/train-A-seed0-q50.log
```

Watch from another terminal with `tail -f runs/train-A-seed0-q50.log`. Ctrl+C stops cleanly
and writes nothing.

## Log line

```
step 1000/1000  loss 0.68555  pearson 0.5061  top10 0.091  query top10 0.107  28803 texts/s
mem 0.01+0.52 GB (peak 0.17)  elapsed 0:00:08  left 0:00:00
```

- `pearson`, `top10`: agreement with the teacher on 2,048 held-out passages
- `query top10`: query-to-passage ranking agreement on 1,024 held-out queries
- `mem active+cache`: MLX memory; the cache should stay flat, capped by `--cache-limit-gb`

## Score

Development decisions use the v2 dev split (`--split train`); val is for the gate only
(ADR-0010).

```sh
uv run tinylab-eval --set data/lumen-v2 run --system student-hybrid \
    --checkpoint checkpoints/student-linear-6144x320-design --split train --notes "..."
```
