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

from ..data.features import PAD_MULTIPLE, FeatureCache, pad_batch, sparse_of
from ..encoder import EncoderSpec
from ..gpu_lock import exclusive_gpu
from ..models.student import CONFIGS, LOSSES, MLX_CACHE_LIMIT, Student

TOP_K = 10
GB = 1e9


def batch_arrays(texts: list[str], spec: EncoderSpec):
    buckets, counts, mask = pad_batch([sparse_of(text, spec) for text in texts], PAD_MULTIPLE)
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


def retrieval_agreement(student_q, student_p, teacher_q, teacher_p) -> float:
    """Top-10 overlap of query-to-passage rankings between student and teacher."""
    s_top = np.argpartition(-(student_q @ student_p.T), TOP_K, axis=1)[:, :TOP_K]
    t_top = np.argpartition(-(teacher_q @ teacher_p.T), TOP_K, axis=1)[:, :TOP_K]
    return round(float(np.mean([len(set(a) & set(b)) for a, b in zip(s_top, t_top)]) / TOP_K), 4)


def read_lines(path: Path, start: int, stop: int) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line)["text"] for index, line in enumerate(handle) if start <= index < stop]


class TextPool:
    """Memory-mapped features and teacher vectors; the last `holdout` rows are never trained on."""

    def __init__(self, path: Path, teacher_path: Path, spec: EncoderSpec, holdout: int):
        self.teacher = np.load(teacher_path, mmap_mode="r")
        self.features = FeatureCache.for_file(path, spec)
        count = min(len(self.features), len(self.teacher))
        self.train_count = count - holdout
        self.held_texts = read_lines(path, self.train_count, count)
        self.held_teacher = self.targets(np.arange(self.train_count, count))

    def targets(self, rows) -> np.ndarray:
        return np.asarray(self.teacher[rows], dtype=np.float32)


def parse_mix(spec: str) -> tuple[Path, float]:
    """`path:fraction`; the teacher vectors live beside it as `<stem>-teacher.npy`."""
    path, _, fraction = spec.rpartition(":")
    if not path:
        raise ValueError(f"--mix expects PATH:FRACTION, got {spec!r}")
    return Path(path), float(fraction)


def teacher_for(path: Path) -> Path:
    return path.with_name(f"{path.stem}-teacher.npy")


def inverse_document_frequency(features: FeatureCache, num_buckets: int) -> np.ndarray:
    """IDF per bucket; a bucket occurs at most once per text in the sparse cache."""
    document_frequency = np.bincount(np.asarray(features.buckets), minlength=num_buckets)
    return (np.log((len(features) + 1) / (document_frequency + 1)) + 1.0).astype(np.float32)


def lexical_vectors(sparse, idf: np.ndarray) -> np.ndarray:
    """L2-normalized IDF-weighted signed counts, the ngram-tfidf baseline's vectors."""
    out = np.zeros((len(sparse), len(idf)), dtype=np.float32)
    for row, (buckets, counts) in enumerate(sparse):
        out[row, buckets] = counts * idf[buckets]
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-8)


def memory_line() -> str:
    return (
        f"mem {mx.get_active_memory() / GB:.2f}+{mx.get_cache_memory() / GB:.2f} GB"
        f" (peak {mx.get_peak_memory() / GB:.2f})"
    )


def clock(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes // 60}:{minutes % 60:02d}:{seconds:02d}"


def train(args: argparse.Namespace) -> dict:
    config = CONFIGS[args.config]
    spec = EncoderSpec(num_buckets=config.num_buckets)
    mx.set_cache_limit(int(args.cache_limit_gb * GB))

    passages = TextPool(args.passages, args.teacher, spec, args.holdout)
    queries = None
    if args.queries:
        queries = TextPool(args.queries, args.query_teacher, spec, args.query_holdout)
    n_query = round(args.batch_size * args.query_fraction) if queries else 0
    mixes = [
        (path, TextPool(path, teacher_for(path), spec, 0), round(args.batch_size * fraction))
        for path, fraction in map(parse_mix, args.mix)
    ]
    n_mix = sum(count for _, _, count in mixes)
    if n_query + n_mix >= args.batch_size:
        raise ValueError("queries and --mix fractions must leave room for MS MARCO passages")
    idf = inverse_document_frequency(passages.features, spec.num_buckets) if args.lexical_weight else None
    print(
        f"{passages.train_count:,} passages (+{args.holdout:,} held out)"
        + (f", {queries.train_count:,} queries, {n_query}/{args.batch_size} per batch" if queries else "")
        + "".join(f", {pool.train_count:,} {path.stem}, {count}/{args.batch_size} per batch" for path, pool, count in mixes)
        + f", config {args.config}, loss {args.loss}, lexical weight {args.lexical_weight}"
        + f", {args.steps:,} steps",
        flush=True,
    )
    print(f"  {config.describe()}", flush=True)

    mx.random.seed(args.seed)
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
        rows = rng.integers(0, passages.train_count, size=args.batch_size - n_query - n_mix)
        sparse, target = passages.features.sparse(rows), passages.targets(rows)
        if n_query:
            query_rows = rng.integers(0, queries.train_count, size=n_query)
            sparse += queries.features.sparse(query_rows)
            target = np.concatenate([target, queries.targets(query_rows)])
        for _, pool, count in mixes:
            mix_rows = rng.integers(0, pool.train_count, size=count)
            sparse += pool.features.sparse(mix_rows)
            target = np.concatenate([target, pool.targets(mix_rows)])
        buckets, counts, mask = (mx.array(part) for part in pad_batch(sparse, PAD_MULTIPLE))
        similarity = target @ target.T
        if idf is not None:
            lexical = lexical_vectors(sparse, idf)
            similarity = args.lexical_weight * (lexical @ lexical.T) + (1 - args.lexical_weight) * similarity

        loss, grads = step_fn(model, buckets, counts, mask, mx.array(similarity))
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        value = float(loss)
        if not np.isfinite(value):
            raise FloatingPointError(f"loss became {value} at step {step}; stopping without saving")
        running = value if running is None else 0.98 * running + 0.02 * value
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            held_passages = embed(model, passages.held_texts, spec)
            held = fidelity(held_passages, passages.held_teacher)
            if queries:
                held["query_top10"] = retrieval_agreement(
                    embed(model, queries.held_texts, spec),
                    held_passages,
                    queries.held_teacher,
                    passages.held_teacher,
                )
            elapsed = time.time() - start
            print(
                f"  step {step:>6}/{args.steps}  loss {running:.5f}"
                f"  pearson {held['pearson']:.4f}  top10 {held['top10_overlap']:.3f}"
                + (f"  query top10 {held['query_top10']:.3f}" if queries else "")
                + f"  {step * args.batch_size / elapsed:6.0f} texts/s  {memory_line()}"
                + f"  elapsed {clock(elapsed)}  left {clock(elapsed / step * (args.steps - step))}",
                flush=True,
            )
            history.append({"step": step, "loss": running, **held})

    return {
        "model": model,
        "config": config,
        "history": history,
        "final_loss": running,
        "fidelity": history[-1],
        "seconds": round(time.time() - start, 1),
        "peak_memory_gb": round(mx.get_peak_memory() / GB, 3),
    }


def save(result: dict, args: argparse.Namespace) -> Path:
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"student-{args.config}-seed{args.seed}" + (f"-{args.tag}" if args.tag else "")
    weights = args.out / f"{stem}.safetensors"
    partial = args.out / f"{stem}.partial.safetensors"
    mx.save_safetensors(str(partial), dict(tree_flatten(result["model"].parameters())))
    partial.rename(weights)
    meta = {
        "config": result["config"].to_dict(),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "loss": args.loss,
        "temperature": args.temperature if args.loss == "listwise" else None,
        "lexical_weight": args.lexical_weight,
        "holdout": args.holdout,
        "queries": str(args.queries) if args.queries else None,
        "query_fraction": args.query_fraction if args.queries else 0.0,
        "mix": [[str(path), fraction] for path, fraction in map(parse_mix, args.mix)],
        "fidelity": result["fidelity"],
        "seed": args.seed,
        "seconds": result["seconds"],
        "peak_memory_gb": result["peak_memory_gb"],
        "final_loss": result["final_loss"],
        "history": result["history"],
    }
    (args.out / f"{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return weights


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
    parser.add_argument("--queries", type=Path, help="mix these queries into every batch")
    parser.add_argument("--query-teacher", type=Path, default=Path("data/train/queries-teacher.npy"))
    parser.add_argument("--query-fraction", type=float, default=0.5)
    parser.add_argument("--query-holdout", type=int, default=1024)
    parser.add_argument(
        "--mix", action="append", default=[], metavar="PATH:FRACTION",
        help="share of each batch drawn from another labelled jsonl; repeatable",
    )
    parser.add_argument(
        "--lexical-weight", type=float, default=0.0, help="share of n-gram TF-IDF in the target similarity"
    )
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--cache-limit-gb", type=float, default=MLX_CACHE_LIMIT / GB)
    parser.add_argument("--tag", default="", help="suffix that keeps checkpoint stems unique")
    args = parser.parse_args(argv)

    with exclusive_gpu(f"tinylab-train config {args.config} seed {args.seed} {args.tag}".strip()):
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
