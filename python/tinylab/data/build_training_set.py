"""Sample and deduplicate MS MARCO passages or queries for teacher labelling. See ADR-0007."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

REPO = "BeIR/msmarco"
PARQUETS = {
    "passages": "corpus/corpus-00000-of-00001.parquet",
    "queries": "queries/queries-00000-of-00001.parquet",
}
MIN_WORDS = {"passages": 10, "queries": 2}
MAX_CHARS = 1200
WHITESPACE_RE = re.compile(r"\s+")


def fingerprint(text: str) -> str:
    normalized = WHITESPACE_RE.sub(" ", text.lower()).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def download(kind: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO, PARQUETS[kind], repo_type="dataset"))


def sample(path: Path, count: int, seed: int, min_words: int = MIN_WORDS["passages"]) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["_id", "text"])
    texts = table.column("text").to_pylist()
    ids = table.column("_id").to_pylist()
    print(f"  loaded {len(texts):,} rows")

    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)

    kept: list[dict] = []
    seen: set[str] = set()
    for index in order:
        text = WHITESPACE_RE.sub(" ", texts[index]).strip()[:MAX_CHARS]
        if len(text.split()) < min_words:
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
    parser.add_argument("--kind", default="passages", choices=sorted(PARQUETS))
    parser.add_argument("--out", type=Path, default=None, help="default: data/train/<kind>.jsonl")
    parser.add_argument("--count", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    out = args.out or Path(f"data/train/{args.kind}.jsonl")
    print(f"downloading {REPO}/{PARQUETS[args.kind]}")
    path = download(args.kind)
    print(f"  cached at {path} ({path.stat().st_size / 1e9:.2f} GB)")

    passages = sample(path, args.count, args.seed, MIN_WORDS[args.kind])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in passages:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    words = sum(len(record["text"].split()) for record in passages) / len(passages)
    print(f"wrote {len(passages):,} {args.kind} to {out} (mean {words:.0f} words, seed {args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
