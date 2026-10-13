"""Design-document training text: sections of cloned markdown repositories, parsed like the Lumen eval set."""

from __future__ import annotations

import argparse
import random
import re
import subprocess
from collections import Counter
from pathlib import Path

from .build_lumen_set import MIN_BODY_WORDS, is_usable_query, parse_sections, query_key, query_text
from .build_technical_set import write
from .build_training_set import fingerprint

TEMPLATE_RE = re.compile(r"\{\{.*?\}\}|\{%.*?%\}|\{#[^}]*\}", re.S)
SPACES_RE = re.compile(r"\s+")


def strip_templates(text: str) -> str:
    """Drop Hugo shortcodes, Liquid tags and heading anchors, which are markup, not prose."""
    return SPACES_RE.sub(" ", TEMPLATE_RE.sub(" ", text)).strip()


def git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip() or "unknown"


def sections_of(root: Path) -> list[tuple[str, str]]:
    """(heading, body) for every markdown section under root."""
    found = []
    for path in sorted(root.rglob("*.md")):
        unit = f"{root.name}/{path.relative_to(root).as_posix()}"
        for section in parse_sections(path, unit):
            found.append((strip_templates(section.heading), strip_templates(section.body)))
    return found


def unique_bodies(sections: list[tuple[str, str]]) -> list[str]:
    kept, seen = [], set()
    for _, body in sections:
        digest = fingerprint(body)
        if len(body.split()) >= MIN_BODY_WORDS and digest not in seen:
            seen.add(digest)
            kept.append(body)
    return kept


def unique_titles(sections: list[tuple[str, str]]) -> list[str]:
    """Headings that occur once across all sources and would pass as a Lumen query."""
    counts = Counter(query_key(heading) for heading, _ in sections)
    return sorted(
        {
            query_text(heading)
            for heading, _ in sections
            if counts[query_key(heading)] == 1 and is_usable_query(query_text(heading))
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tinylab-build-design-set", description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path, help="cloned repositories (licences checked)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--name", default="design", help="writes <name>-passages.jsonl, <name>-titles.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    sections = []
    for root in args.sources:
        found = sections_of(root)
        print(f"  {root.name} @ {git_head(root)}: {len(found):,} sections", flush=True)
        sections += found

    bodies, titles = unique_bodies(sections), unique_titles(sections)
    random.Random(args.seed).shuffle(bodies)
    random.Random(args.seed).shuffle(titles)
    for kind, texts in (("passages", bodies), ("titles", titles)):
        records = [{"doc_id": f"{kind[0]}{index}", "text": text} for index, text in enumerate(texts)]
        write(args.out_dir / f"{args.name}-{kind}.jsonl", records)
    print(f"wrote {len(bodies):,} passages and {len(titles):,} titles as {args.out_dir}/{args.name}-*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
