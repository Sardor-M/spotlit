"""Embed the training passages with the teacher and cache the vectors. See ADR-0007."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..eval.dense import DEFAULT_TEACHER


def load_texts(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line)["text"] for line in handle if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tinylab-label", description=__doc__)
    parser.add_argument("--passages", type=Path, default=Path("data/train/passages.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/train/teacher.npy"))
    parser.add_argument("--model", default=DEFAULT_TEACHER)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="label only the first N passages")
    args = parser.parse_args(argv)

    from sentence_transformers import SentenceTransformer

    texts = load_texts(args.passages)
    if args.limit:
        texts = texts[: args.limit]
    print(f"labelling {len(texts):,} passages with {args.model}")

    model = SentenceTransformer(args.model)
    vectors = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float16)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, vectors)
    print(f"wrote {args.out} {vectors.shape} ({vectors.nbytes / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
