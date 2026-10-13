"""Technical StackExchange posts (CQADupStack) as documentation-shaped training text. See ADR-0007."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .build_training_set import MAX_CHARS, MIN_WORDS, WHITESPACE_RE, fingerprint

FORUMS = ("unix", "programmers", "webmasters", "wordpress", "android", "gis")
REPO = "mteb/cqadupstack-{forum}"


def load_forum(forum: str) -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO.format(forum=forum), "corpus.jsonl", repo_type="dataset")
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def clean(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text or "").strip()[:MAX_CHARS]


def pairs(forums: tuple[str, ...], seed: int) -> list[tuple[str, str, str]]:
    """(doc_id, title, body) per post, shuffled, deduplicated by body."""
    rows = []
    for forum in forums:
        posts = load_forum(forum)
        print(f"  {forum}: {len(posts):,} posts", flush=True)
        rows += [(f"{forum}/{post['_id']}", clean(post.get("title")), clean(post.get("text"))) for post in posts]
    random.Random(seed).shuffle(rows)

    kept, seen = [], set()
    for doc_id, title, body in rows:
        if len(body.split()) < MIN_WORDS["passages"] or len(title.split()) < MIN_WORDS["queries"]:
            continue
        digest = fingerprint(body)
        if digest in seen:
            continue
        seen.add(digest)
        kept.append((doc_id, title, body))
    return kept


def write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tinylab-build-technical-set", description=__doc__)
    parser.add_argument("--forums", default=",".join(FORUMS))
    parser.add_argument("--out-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--name", default="tech", help="writes <name>-passages.jsonl and <name>-titles.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    forums = tuple(name.strip() for name in args.forums.split(",") if name.strip())
    print(f"downloading CQADupStack: {', '.join(forums)}", flush=True)
    kept = pairs(forums, args.seed)
    write(args.out_dir / f"{args.name}-passages.jsonl", [{"doc_id": d, "text": body} for d, _, body in kept])
    write(args.out_dir / f"{args.name}-titles.jsonl", [{"doc_id": d, "text": title} for d, title, _ in kept])

    words = sum(len(body.split()) for _, _, body in kept) / max(len(kept), 1)
    print(f"wrote {len(kept):,} body/title pairs to {args.out_dir} (mean body {words:.0f} words, seed {args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
