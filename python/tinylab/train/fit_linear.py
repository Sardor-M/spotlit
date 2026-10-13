"""Closed-form linear student: ridge fit of the n-gram table to the teacher's top principal components."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..encoder import EncoderSpec
from ..gpu_lock import exclusive_gpu
from ..models.student import StudentConfig
from .distill import TextPool, teacher_for

TEACHER_DIM = 384
ROWS_PER_CHUNK = 40_000_000


def pooled(pool: TextPool, rows: np.ndarray, buckets: int) -> np.ndarray:
    """Count-weighted mean over present buckets, the student's own pooling."""
    out = np.zeros((len(rows), buckets))
    for row, (ids, counts) in enumerate(pool.features.sparse(rows)):
        out[row, ids] = counts / max(len(ids), 1)
    return out


def accumulate(pools: list[TextPool], buckets: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    gram = np.zeros((buckets, buckets))
    cross = np.zeros((buckets, TEACHER_DIM))
    covariance = np.zeros((TEACHER_DIM, TEACHER_DIM))
    chunk = max(1000, ROWS_PER_CHUNK // buckets)
    total = 0
    for pool in pools:
        for start in range(0, pool.train_count, chunk):
            rows = np.arange(start, min(start + chunk, pool.train_count))
            features = pooled(pool, rows, buckets)
            targets = pool.targets(rows).astype(np.float64)
            gram += features.T @ features
            cross += features.T @ targets
            covariance += targets.T @ targets
        total += pool.train_count
    return gram, cross, covariance, total


def fit(args: argparse.Namespace) -> tuple[np.ndarray, StudentConfig, dict]:
    spec = EncoderSpec(num_buckets=args.buckets)
    components = args.components or min(TEACHER_DIM, args.budget // args.buckets)
    pools = [TextPool(path, teacher_for(path), spec, 0) for path in args.source]
    start = time.time()
    gram, cross, covariance, rows = accumulate(pools, args.buckets)
    _, eigenvectors = np.linalg.eigh(covariance)
    basis = eigenvectors[:, ::-1][:, :components]
    regularized = gram + args.ridge * np.trace(gram) / args.buckets * np.eye(args.buckets)
    table = np.linalg.solve(regularized, cross @ basis).astype(np.float32)
    config = StudentConfig(num_buckets=args.buckets, d_table=components, hidden=0, d_out=components)
    meta = {"rows": rows, "seconds": round(time.time() - start, 1), "basis": basis.astype(np.float32)}
    return table, config, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tinylab-fit", description=__doc__)
    parser.add_argument("--buckets", type=int, default=6144)
    parser.add_argument("--components", type=int, default=0, help="0 = as many as the budget allows")
    parser.add_argument("--budget", type=int, default=1_966_080, help="table parameters, ADR-0008")
    parser.add_argument(
        "--source", type=Path, action="append", default=[], help="labelled jsonl; repeatable"
    )
    parser.add_argument("--ridge", type=float, default=1e-3, help="relative to the mean feature energy")
    parser.add_argument("--out", type=Path, default=Path("checkpoints"))
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)
    if not args.source:
        args.source = [Path("data/train/tech-passages.jsonl"), Path("data/train/tech-titles.jsonl")]

    import mlx.core as mx

    with exclusive_gpu(f"tinylab-fit {args.buckets} buckets"):
        table, config, meta = fit(args)
    stem = f"student-linear-{config.num_buckets}x{config.d_table}" + (f"-{args.tag}" if args.tag else "")
    args.out.mkdir(parents=True, exist_ok=True)
    partial = args.out / f"{stem}.partial.safetensors"
    mx.save_safetensors(str(partial), {"table.weight": mx.array(table)})
    partial.rename(args.out / f"{stem}.safetensors")
    np.save(args.out / f"{stem}.basis.npy", meta.pop("basis"))
    (args.out / f"{stem}.json").write_text(
        json.dumps(
            {
                "config": config.to_dict(),
                "method": "ridge on teacher principal components",
                "sources": [str(path) for path in args.source],
                "ridge": args.ridge,
                **meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"saved {args.out / stem}.safetensors  {config.describe()}  "
        f"fit on {meta['rows']:,} texts in {meta['seconds']:.0f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
