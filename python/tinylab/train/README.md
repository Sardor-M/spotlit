# tinylab.train

Distills the teacher into the student (ADR-0006). One GPU job at a time: a second
`tinylab-train` or `tinylab-label` refuses to start while one is running.

## Run

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

```sh
uv run tinylab-eval --set data/lumen run --system student-hybrid \
    --checkpoint checkpoints/student-A-seed0-q50 --split val --notes "..."
```
