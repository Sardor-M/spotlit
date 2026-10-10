"""Sample and deduplicate MS MARCO passages for teacher labelling. See ADR-0007."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

REPO = "BeIR/msmarco"
PARQUET = "corpus/corpus-00000-of-00001.parquet"
MIN_WORDS = 10
MAX_CHARS = 1200
WHITESPACE_RE = re.compile(r"\s+")


def fingerprint(text: str) -> str:
    normalized = WHITESPACE_RE.sub(" ", text.lower()).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def download() -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO, PARQUET, repo_type="dataset"))


def sample(path: Path, count: int, seed: int) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["_id", "text"])
    texts = table.column("text").to_pylist()
    ids = table.column("_id").to_pylist()
    print(f"  loaded {len(texts):,} passages")

    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)

    kept: list[dict] = []
    seen: set[str] = set()
    for index in order:
        text = WHITESPACE_RE.sub(" ", texts[index]).strip()[:MAX_CHARS]
        if len(text.split()) < MIN_WORDS:
            continue
        digest = fingerprint(text)
        if digest in seen:
            continue
        seen.add(digest)
        kept.append({"doc_id": str(ids[index]), "text": text})
        if len(kept) >= count:
            break
    return kept


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tinylab-build-training-set", description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/train/passages.jsonl"))
    parser.add_argument("--count", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    print(f"downloading {REPO}/{PARQUET}")
    path = download()
    print(f"  cached at {path} ({path.stat().st_size / 1e9:.2f} GB)")

    passages = sample(path, args.count, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in passages:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    words = sum(len(record["text"].split()) for record in passages) / len(passages)
    print(f"wrote {len(passages):,} passages to {args.out} (mean {words:.0f} words, seed {args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
