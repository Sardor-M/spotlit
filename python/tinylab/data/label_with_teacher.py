"""Embed the training passages with the teacher and cache the vectors. See ADR-0007."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..eval.dense import DEFAULT_TEACHER, QUERY_PREFIX
from ..gpu_lock import exclusive_gpu

# PyTorch's default MPS cache may reach 1.7x the recommended working set, enough to freeze a 24 GB Mac.
MPS_MEMORY_FRACTION = 0.4
CHUNK = 8192


def load_texts(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line)["text"] for line in handle if line.strip()]


def limit_mps_memory(torch) -> bool:
    if not torch.backends.mps.is_available():
        return False
    torch.mps.set_per_process_memory_fraction(MPS_MEMORY_FRACTION)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tinylab-label", description=__doc__)
    parser.add_argument("--passages", type=Path, default=Path("data/train/passages.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/train/teacher.npy"))
    parser.add_argument("--model", default=DEFAULT_TEACHER)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="label only the first N passages")
    parser.add_argument("--query-prefix", action="store_true", help="embed as retrieval queries")
    args = parser.parse_args(argv)

    import torch
    from sentence_transformers import SentenceTransformer

    texts = load_texts(args.passages)
    if args.limit:
        texts = texts[: args.limit]
    print(f"labelling {len(texts):,} passages with {args.model}")

    prefix = QUERY_PREFIX if args.query_prefix else ""
    with exclusive_gpu(f"tinylab-label {args.passages}"):
        mps = limit_mps_memory(torch)
        model = SentenceTransformer(args.model)
        chunks, start = [], time.time()
        for first in range(0, len(texts), CHUNK):
            window = [prefix + text for text in texts[first : first + CHUNK]]
            chunks.append(
                model.encode(
                    window,
                    batch_size=args.batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                ).astype(np.float16)
            )
            if mps:
                torch.mps.empty_cache()
            done = min(first + CHUNK, len(texts))
            elapsed = time.time() - start
            print(
                f"  {done:>9,}/{len(texts):,}  {done / elapsed:5.0f} texts/s"
                f"  left {(len(texts) - done) / max(done / elapsed, 1e-9) / 60:4.1f} min"
                + (f"  gpu {torch.mps.driver_allocated_memory() / 1e9:.1f} GB" if mps else ""),
                flush=True,
            )
        vectors = np.concatenate(chunks)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.stem + ".partial.npy")
    np.save(partial, vectors)
    partial.rename(args.out)
    print(f"wrote {args.out} {vectors.shape} ({vectors.nbytes / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
