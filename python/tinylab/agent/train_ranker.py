"""Train the 2 MB element ranker: the linear hashed table, softmax over each page's candidates. See ADR-0012."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from ..data.features import sparse_of
from ..encoder import EncoderSpec
from ..gpu_lock import exclusive_gpu
from ..models.student import MLX_CACHE_LIMIT, Student, StudentConfig
from ..train.distill import GB, clock, memory_line
from .rank import KS, load_steps, query_of

# Text counts rounded up to this, so MLX sees few distinct batch shapes.
TEXT_MULTIPLE = 256
MASKED = -1e9
PAD_TEXT = "padding"


def step_arrays(step: dict, spec: EncoderSpec):
    """Pooled bucket counts, query first then candidates, as the closed-form fit pools them."""
    texts = [query_of(step)] + [text for _, text in step["candidates"]]
    rows = -(-len(texts) // TEXT_MULTIPLE) * TEXT_MULTIPLE
    # Padding rows get a real text: an all-zero row has an undefined gradient through the L2 norm.
    texts += [PAD_TEXT] * (rows - len(texts))
    pooled = np.zeros((rows, spec.num_buckets), dtype=np.float32)
    for row, text in enumerate(texts):
        buckets, counts = sparse_of(text, spec)
        pooled[row, buckets] = counts / max(len(buckets), 1)
    gold = set(step["gold_described"])
    valid = np.zeros(rows - 1, dtype=bool)
    valid[: len(step["candidates"])] = True
    is_gold = np.zeros(rows - 1, dtype=bool)
    is_gold[: len(step["candidates"])] = [node_id in gold for node_id, _ in step["candidates"]]
    return mx.array(pooled), mx.array(valid), mx.array(is_gold)


def embed(model: Student, pooled: mx.array) -> mx.array:
    out = pooled @ model.table.weight
    return out / mx.maximum(mx.linalg.norm(out, axis=-1, keepdims=True), 1e-8)


def ranking_loss(model, pooled, valid, is_gold, temperature):
    vectors = embed(model, pooled)
    scores = mx.where(valid, (vectors[1:] @ vectors[0]) / temperature, MASKED)
    return mx.logsumexp(scores) - mx.logsumexp(mx.where(is_gold, scores, MASKED))


def recall_at(model: Student, steps: list[dict], spec: EncoderSpec) -> dict:
    hits = {k: 0 for k in KS}
    for step in steps:
        pooled, valid, is_gold = step_arrays(step, spec)
        vectors = embed(model, pooled)
        scores = np.array(vectors[1:] @ vectors[0])[: len(step["candidates"])]
        gold = np.array(is_gold)[: len(step["candidates"])]
        rank = int((scores > scores[gold].max()).sum()) + 1 if gold.any() else None
        for k in KS:
            hits[k] += rank is not None and rank <= k
    return {f"recall@{k}": round(hits[k] / len(steps), 4) for k in KS}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tinylab-train-ranker", description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/mind2web/train.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("data/mind2web/dev.jsonl"))
    parser.add_argument("--init", type=Path, default=Path("checkpoints/student-linear-6144x320-design"))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--dev-steps", type=int, default=400, help="dev steps scored at each log line")
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("checkpoints"))
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)

    with exclusive_gpu(f"tinylab-train-ranker {args.tag}".strip()):
        try:
            return train(args)
        except KeyboardInterrupt:
            print("\nstopped by Ctrl+C; no checkpoint written", flush=True)
            return 130


def train(args: argparse.Namespace) -> int:
    mx.set_cache_limit(MLX_CACHE_LIMIT)
    meta = json.loads(args.init.with_name(args.init.name + ".json").read_text(encoding="utf-8"))
    config = StudentConfig(**meta["config"])
    model = Student(config)
    model.update(tree_unflatten(list(mx.load(str(args.init) + ".safetensors").items())))
    spec = EncoderSpec(num_buckets=config.num_buckets)

    steps = [step for step in load_steps(args.train) if step["gold_described"]]
    dev = load_steps(args.dev)
    dev_sample = dev[:: max(1, len(dev) // args.dev_steps)][: args.dev_steps]
    updates = int(args.epochs * len(steps))
    print(f"{len(steps):,} trainable steps, {len(dev):,} dev steps ({len(dev_sample)} scored per log line); "
          f"{updates:,} updates from {args.init.name}; {config.describe()}", flush=True)

    schedule = optim.cosine_decay(args.learning_rate, updates, args.learning_rate / 10)
    optimizer = optim.Adam(learning_rate=schedule)
    step_fn = nn.value_and_grad(model, ranking_loss)
    rng = np.random.default_rng(args.seed)
    print(f"  start  dev {recall_at(model, dev_sample, spec)}", flush=True)

    history, running, start = [], None, time.time()
    order = np.concatenate([rng.permutation(len(steps)) for _ in range(int(np.ceil(args.epochs)))])[:updates]
    for update, index in enumerate(order, 1):
        loss, grads = step_fn(model, *step_arrays(steps[index], spec), args.temperature)
        grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        value = float(loss)
        if not np.isfinite(value):
            raise FloatingPointError(f"loss became {value} at update {update}; stopping without saving")
        running = value if running is None else 0.99 * running + 0.01 * value
        if update % args.log_every == 0 or update == updates:
            report = recall_at(model, dev_sample, spec)
            elapsed = time.time() - start
            print(f"  update {update:>6}/{updates}  loss {running:.4f}  dev {report}  {memory_line()}"
                  f"  elapsed {clock(elapsed)}  left {clock(elapsed / update * (updates - update))}", flush=True)
            history.append({"update": update, "loss": running, **report})

    stem = f"ranker-linear-{config.num_buckets}x{config.d_table}" + (f"-{args.tag}" if args.tag else "")
    args.out.mkdir(parents=True, exist_ok=True)
    partial = args.out / f"{stem}.partial.safetensors"
    mx.save_safetensors(str(partial), dict(tree_flatten(model.parameters())))
    partial.rename(args.out / f"{stem}.safetensors")
    meta = {
        "config": config.to_dict(), "init": str(args.init), "epochs": args.epochs,
        "learning_rate": args.learning_rate, "temperature": args.temperature, "seed": args.seed,
        "history": history, "seconds": round(time.time() - start, 1),
        "peak_memory_gb": round(mx.get_peak_memory() / GB, 3),
    }
    (args.out / f"{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {args.out / stem}.safetensors in {clock(time.time() - start)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
