"""Relational distillation of the teacher's similarity structure. See ADR-0006."""

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

from ..data.features import FeatureCache, pad_batch, sparse_of
from ..data.label_with_teacher import load_texts
from ..encoder import EncoderSpec
from ..models.student import CONFIGS, LOSSES, Student

TOP_K = 10


def batch_arrays(texts: list[str], spec: EncoderSpec):
    buckets, counts, mask = pad_batch([sparse_of(text, spec) for text in texts])
    return mx.array(buckets), mx.array(counts), mx.array(mask)


def embed(model: Student, texts: list[str], spec: EncoderSpec, batch: int = 512) -> np.ndarray:
    chunks = []
    for start in range(0, len(texts), batch):
        out = model(*batch_arrays(texts[start : start + batch], spec))
        mx.eval(out)
        chunks.append(np.array(out))
    return np.concatenate(chunks, axis=0)


def fidelity(student: np.ndarray, teacher: np.ndarray) -> dict:
    """Agreement with the teacher on held-out text: pair correlation and top-10 neighbour overlap."""
    off = ~np.eye(len(teacher), dtype=bool)
    s_sim, t_sim = student @ student.T, teacher @ teacher.T
    pearson = float(np.corrcoef(s_sim[off], t_sim[off])[0, 1])
    s_sim[~off] = t_sim[~off] = -np.inf
    s_top = np.argpartition(-s_sim, TOP_K, axis=1)[:, :TOP_K]
    t_top = np.argpartition(-t_sim, TOP_K, axis=1)[:, :TOP_K]
    overlap = np.mean([len(set(a) & set(b)) for a, b in zip(s_top, t_top)]) / TOP_K
    return {"pearson": round(pearson, 4), "top10_overlap": round(float(overlap), 4)}


def train(args: argparse.Namespace) -> dict:
    config = CONFIGS[args.config]
    spec = EncoderSpec(num_buckets=config.num_buckets)

    texts = load_texts(args.passages)
    teacher = np.load(args.teacher).astype(np.float32)
    if len(texts) > len(teacher):
        texts = texts[: len(teacher)]
    cache_path = args.passages.with_name(f"features-{config.num_buckets}.npz")
    features = FeatureCache.load_or_build(cache_path, texts, spec)
    held_texts, texts = texts[-args.holdout :], texts[: -args.holdout]
    held_teacher = teacher[len(texts) : len(texts) + args.holdout]
    print(f"{len(texts):,} passages (+{args.holdout:,} held out), config {args.config}, loss {args.loss}")
    print(f"  {config.describe()}")

    model = Student(config)
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=args.weight_decay)

    objective = LOSSES[args.loss]
    options = {"temperature": args.temperature} if args.loss == "listwise" else {}

    def loss_fn(model, buckets, counts, mask, target):
        return objective(model(buckets, counts, mask), target, **options)

    step_fn = nn.value_and_grad(model, loss_fn)
    rng = np.random.default_rng(args.seed)

    history = []
    running = None
    start = time.time()
    for step in range(1, args.steps + 1):
        rows = rng.integers(0, len(texts), size=args.batch_size)
        buckets, counts, mask = (mx.array(part) for part in features.batch(rows))
        target = mx.array(teacher[rows])

        loss, grads = step_fn(model, buckets, counts, mask, target)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        value = float(loss)
        running = value if running is None else 0.98 * running + 0.02 * value
        if step % args.log_every == 0 or step == 1:
            held = fidelity(embed(model, held_texts, spec), held_teacher)
            rate = step * args.batch_size / (time.time() - start)
            print(
                f"  step {step:>6}  loss {running:.5f}  held-out pearson {held['pearson']:.4f}"
                f"  top10 {held['top10_overlap']:.3f}  {rate:6.0f} texts/s"
            )
            history.append({"step": step, "loss": running, **held})

    return {
        "model": model,
        "config": config,
        "history": history,
        "final_loss": running,
        "fidelity": history[-1],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tinylab-train", description=__doc__)
    parser.add_argument("--config", default="A", choices=sorted(CONFIGS))
    parser.add_argument("--passages", type=Path, default=Path("data/train/passages.jsonl"))
    parser.add_argument("--teacher", type=Path, default=Path("data/train/teacher.npy"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints"))
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--loss", default="listwise", choices=sorted(LOSSES))
    parser.add_argument("--temperature", type=float, default=0.05, help="listwise loss only")
    parser.add_argument("--holdout", type=int, default=2048, help="last N passages, never trained on")
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--tag", default="", help="suffix that keeps checkpoint stems unique")
    args = parser.parse_args(argv)

    result = train(args)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"student-{args.config}-seed{args.seed}" + (f"-{args.tag}" if args.tag else "")

    weights = dict(tree_flatten(result["model"].parameters()))
    mx.save_safetensors(str(args.out / f"{stem}.safetensors"), weights)
    (args.out / f"{stem}.json").write_text(
        json.dumps(
            {
                "config": result["config"].to_dict(),
                "steps": args.steps,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "loss": args.loss,
                "temperature": args.temperature if args.loss == "listwise" else None,
                "holdout": args.holdout,
                "fidelity": result["fidelity"],
                "seed": args.seed,
                "final_loss": result["final_loss"],
                "history": result["history"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {args.out / stem}.safetensors  final loss {result['final_loss']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
