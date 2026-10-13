"""Distil the teacher into the contextual student by cosine to its vectors. See ADR-0011."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from ..data.words import WordCache, WordSpec, assemble
from ..gpu_lock import exclusive_gpu
from ..models.contextual import SIZES, ContextualStudent, embed
from ..models.student import MLX_CACHE_LIMIT
from .distill import GB, clock, fidelity, memory_line, read_lines, retrieval_agreement, teacher_for

KINDS = ("passages", "titles")
SOURCES = [Path(f"data/train/{name}-{kind}.jsonl") for name in ("tech", "stats", "design") for kind in KINDS]
HELD_PASSAGES, HELD_TITLES = Path("data/train/design-passages.jsonl"), Path("data/train/design-titles.jsonl")
HOLDOUT = {HELD_PASSAGES: 2048, HELD_TITLES: 1024}
SOURCE_SHIFT = 32


class Source:
    """A labelled jsonl: word cache, teacher vectors, and its trainable leading rows."""

    def __init__(self, path: Path, spec: WordSpec, holdout: int):
        self.path = path
        self.cache = WordCache.for_file(path, spec)
        self.teacher = np.load(teacher_for(path), mmap_mode="r")
        count = min(len(self.cache), len(self.teacher))
        self.train_count = count - holdout
        self.held_texts = read_lines(path, self.train_count, count)
        self.held_teacher = np.asarray(self.teacher[self.train_count : count], dtype=np.float32)


def sample_batch(sources: list[Source], picks: np.ndarray, starts: np.ndarray, spec: WordSpec):
    """Word batch and teacher targets for global row numbers `picks`."""
    owners = np.searchsorted(starts, picks, side="right") - 1
    sequences, targets = [], []
    for owner, pick in zip(owners, picks):
        source, row = sources[owner], int(pick - starts[owner])
        sequences.append(source.cache.sequence(row) | (int(owner) << SOURCE_SHIFT))
        targets.append(source.teacher[row])
    mask = (1 << SOURCE_SHIFT) - 1
    batch = assemble(
        sequences, lambda unique: [sources[key >> SOURCE_SHIFT].cache.grams(key & mask) for key in unique], spec
    )
    targets = np.asarray(targets, dtype=np.float32)
    return batch, targets / np.maximum(np.linalg.norm(targets, axis=1, keepdims=True), 1e-8)


def train(args: argparse.Namespace) -> dict:
    config = SIZES[args.size]
    spec = config.word_spec()
    mx.set_cache_limit(int(args.cache_limit_gb * GB))

    sources = [Source(path, spec, HOLDOUT.get(path, 0)) for path in args.source]
    counts = np.array([source.train_count for source in sources])
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    held = {source.path: source for source in sources}
    print(
        ", ".join(f"{source.train_count:,} {source.path.stem}" for source in sources)
        + f"; size {args.size}, {args.steps:,} steps of {args.batch_size}",
        flush=True,
    )
    print(f"  {config.describe()}", flush=True)

    mx.random.seed(args.seed)
    model = ContextualStudent(config)
    warmup = min(args.warmup, args.steps // 2)
    schedule = optim.join_schedules(
        [
            optim.linear_schedule(args.learning_rate / 100, args.learning_rate, warmup),
            optim.cosine_decay(args.learning_rate, args.steps - warmup, args.learning_rate / 10),
        ],
        [warmup],
    )
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=args.weight_decay)

    def loss_fn(model, grams, gram_mask, words, word_mask, target):
        return 1.0 - (model(grams, gram_mask, words, word_mask) * target).sum(axis=-1).mean()

    step_fn = nn.value_and_grad(model, loss_fn)
    rng = np.random.default_rng(args.seed)
    history, running, start = [], None, time.time()
    for step in range(1, args.steps + 1):
        batch, target = sample_batch(sources, rng.integers(0, counts.sum(), size=args.batch_size), starts, spec)
        loss, grads = step_fn(
            model, *(mx.array(part) for part in (batch.grams, batch.gram_mask, batch.words, batch.word_mask)),
            mx.array(target),
        )
        grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        value = float(loss)
        if not np.isfinite(value):
            raise FloatingPointError(f"loss became {value} at step {step}; stopping without saving")
        running = value if running is None else 0.98 * running + 0.02 * value
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            passages, titles = held.get(HELD_PASSAGES), held.get(HELD_TITLES)
            report = {}
            if passages is not None:
                student_passages = embed(model, passages.held_texts)
                report = fidelity(student_passages, passages.held_teacher)
                if titles is not None:
                    student_titles = embed(model, titles.held_texts)
                    report["query_top10"] = retrieval_agreement(
                        student_titles, student_passages, titles.held_teacher, passages.held_teacher
                    )
            elapsed = time.time() - start
            print(
                f"  step {step:>6}/{args.steps}  loss {running:.5f}"
                + "".join(f"  {name} {value:.4f}" for name, value in report.items())
                + f"  {step * args.batch_size / elapsed:6.0f} texts/s  {memory_line()}"
                + f"  elapsed {clock(elapsed)}  left {clock(elapsed / step * (args.steps - step))}",
                flush=True,
            )
            history.append({"step": step, "loss": running, **report})

    return {
        "model": model,
        "config": config,
        "history": history,
        "final_loss": running,
        "seconds": round(time.time() - start, 1),
        "peak_memory_gb": round(mx.get_peak_memory() / GB, 3),
    }


def save(result: dict, args: argparse.Namespace) -> Path:
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"student-ctx-{args.size}-seed{args.seed}" + (f"-{args.tag}" if args.tag else "")
    partial = args.out / f"{stem}.partial.safetensors"
    mx.save_safetensors(str(partial), dict(tree_flatten(result["model"].parameters())))
    partial.rename(args.out / f"{stem}.safetensors")
    meta = {
        "config": result["config"].to_dict(),
        "words": result["config"].word_spec().to_dict(),
        "sources": [str(path) for path in args.source],
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup": args.warmup,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "seconds": result["seconds"],
        "peak_memory_gb": result["peak_memory_gb"],
        "final_loss": result["final_loss"],
        "history": result["history"],
    }
    (args.out / f"{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return args.out / f"{stem}.safetensors"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tinylab-train-contextual", description=__doc__)
    parser.add_argument("--size", default="S", choices=sorted(SIZES))
    parser.add_argument("--source", type=Path, action="append", default=[], help="labelled jsonl; repeatable")
    parser.add_argument("--out", type=Path, default=Path("checkpoints"))
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--cache-limit-gb", type=float, default=MLX_CACHE_LIMIT / GB)
    parser.add_argument("--tag", default="", help="suffix that keeps checkpoint stems unique")
    args = parser.parse_args(argv)
    args.source = args.source or SOURCES

    with exclusive_gpu(f"tinylab-train-contextual size {args.size} seed {args.seed} {args.tag}".strip()):
        try:
            result = train(args)
        except KeyboardInterrupt:
            print("\nstopped by Ctrl+C; no checkpoint written", flush=True)
            return 130
        weights = save(result, args)
    print(
        f"saved {weights}  final loss {result['final_loss']:.5f}"
        f"  in {clock(result['seconds'])}, peak MLX memory {result['peak_memory_gb']:.2f} GB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
