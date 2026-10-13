"""Build a retrieval evaluation set from a markdown corpus. See ADR-0004."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

MIN_BODY_WORDS = 40
MAX_BODY_CHARS = 1200
MIN_HEADING_WORDS = 2
MAX_HEADING_WORDS = 12
NEAR_DUPLICATE_JACCARD = 0.7

# Headings too generic to have a single correct answer across a whole corpus.
GENERIC_HEADINGS = {
    "abstract", "api", "appendix", "architecture", "background", "changelog",
    "conclusion", "configuration", "context", "contributing", "decision", "design",
    "examples", "faq", "future work", "getting started", "glossary", "goals",
    "implementation", "installation", "introduction", "license", "licence",
    "motivation", "next steps", "notes", "open questions", "overview",
    "prerequisites", "problem", "references", "requirements", "results", "risks",
    "roadmap", "scope", "solution", "status", "summary", "table of contents",
    "testing", "tests", "tldr", "tl;dr", "todo", "usage", "why", "how it works",
}

FENCE_RE = re.compile(r"^\s*(```|~~~)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
HTML_TAG_RE = re.compile(r"<[^>]+>")
LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
EMPHASIS_RE = re.compile(r"[*_`]+")
WORD_RE = re.compile(r"[a-z0-9]+")
NUMBER_PREFIX_RE = re.compile(r"^\d+(?:\.\d+)*[.)]?\s+")


@dataclass
class Section:
    unit_id: str
    doc_id: str
    level: int
    heading: str
    body: str
    descendants: list[str] = field(default_factory=list)


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "section"


def clean_prose(lines: list[str]) -> str:
    """Strip markdown to plain prose, dropping code, tables and HTML."""
    kept = []
    in_fence = False
    for line in lines:
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("|") or set(stripped) <= set("-=|: "):
            continue
        stripped = LIST_MARKER_RE.sub("", stripped)
        stripped = stripped.lstrip("> ").strip()
        kept.append(stripped)

    text = " ".join(kept)
    text = HTML_COMMENT_RE.sub(" ", text)
    text = IMAGE_RE.sub(" ", text)
    text = LINK_RE.sub(r"\1", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = EMPHASIS_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def truncate(text: str, limit: int = MAX_BODY_CHARS) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    stop = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[: stop + 1].strip() if stop > limit // 2 else cut.rstrip() + "..."


def parse_sections(path: Path, unit_id: str) -> list[Section]:
    """One section per heading; body runs until the next heading of any level."""
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()

    entries: list[tuple[int, str, list[str]]] = []
    in_fence = False
    for line in raw:
        if FENCE_RE.match(line):
            in_fence = not in_fence
            if entries:
                entries[-1][2].append(line)
            continue
        match = None if in_fence else HEADING_RE.match(line)
        if match:
            entries.append((len(match.group(1)), match.group(2).strip(), []))
        elif entries:
            entries[-1][2].append(line)

    sections = []
    seen_slugs: dict[str, int] = {}
    for level, heading, body_lines in entries:
        slug = slugify(heading)
        seen_slugs[slug] = seen_slugs.get(slug, 0) + 1
        if seen_slugs[slug] > 1:
            slug = f"{slug}-{seen_slugs[slug]}"
        sections.append(
            Section(
                unit_id=unit_id,
                doc_id=f"{unit_id}#{slug}",
                level=level,
                heading=heading,
                body=truncate(clean_prose(body_lines)),
            )
        )

    for index, section in enumerate(sections):
        for later in sections[index + 1 :]:
            if later.level <= section.level:
                break
            section.descendants.append(later.doc_id)
    return sections


def tokens(text: str) -> set[str]:
    return {word for word in WORD_RE.findall(text.lower()) if len(word) > 2}


def query_text(heading: str) -> str:
    """Heading minus any leading section number, which is a label not a question."""
    return NUMBER_PREFIX_RE.sub("", heading).strip(" :-—")


def query_key(heading: str) -> str:
    """Identity of the query a heading becomes, so numbered repeats count as one."""
    return query_text(heading).lower().strip(" :?")


def is_usable_query(heading: str) -> bool:
    words = heading.split()
    if not MIN_HEADING_WORDS <= len(words) <= MAX_HEADING_WORDS:
        return False
    if heading.lower().strip(" :?") in GENERIC_HEADINGS:
        return False
    return len([word for word in words if len(word) >= 3 and word[0].isalpha()]) >= 2


def collect(source: Path, skip_patterns: tuple[str, ...]) -> list[Section]:
    sections: list[Section] = []
    for path in sorted(source.rglob("*.md")):
        relative = path.relative_to(source).as_posix()
        if any(pattern in relative for pattern in skip_patterns):
            continue
        sections.extend(parse_sections(path, relative))
    return sections


def drop_near_duplicates(sections: list[Section]) -> tuple[list[Section], int]:
    """Keep the first of any group of near-identical passages."""
    kept: list[Section] = []
    kept_tokens: list[set[str]] = []
    dropped = 0
    for section in sections:
        current = tokens(section.body)
        duplicate = False
        for existing in kept_tokens:
            union = len(current | existing)
            if union and len(current & existing) / union >= NEAR_DUPLICATE_JACCARD:
                duplicate = True
                break
        if duplicate:
            dropped += 1
            continue
        kept.append(section)
        kept_tokens.append(current)
    return kept, dropped


def build(source: Path, out: Path, skip_patterns: tuple[str, ...]) -> dict:
    sections = collect(source, skip_patterns)
    with_body = [s for s in sections if len(s.body.split()) >= MIN_BODY_WORDS]
    deduped, n_duplicates = drop_near_duplicates(with_body)

    by_id = {section.doc_id: section for section in deduped}
    heading_counts: dict[str, int] = {}
    for section in deduped:
        key = query_key(section.heading)
        heading_counts[key] = heading_counts.get(key, 0) + 1

    queries = []
    for section in deduped:
        text = query_text(section.heading)
        if heading_counts[query_key(section.heading)] > 1 or not is_usable_query(text):
            continue
        relevant = {section.doc_id: 2}
        for doc_id in section.descendants:
            if doc_id in by_id:
                relevant[doc_id] = 1
        queries.append(
            {
                "query_id": f"h-{slugify(text)}-{len(queries):03d}",
                "text": text,
                "relevant": relevant,
            }
        )

    out.mkdir(parents=True, exist_ok=True)
    with (out / "corpus.jsonl").open("w", encoding="utf-8") as handle:
        for section in deduped:
            record = {"doc_id": section.doc_id, "unit_id": section.unit_id, "text": section.body}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (out / "queries.jsonl").open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(json.dumps(query, ensure_ascii=False) + "\n")

    return {
        "source": str(source),
        "files": len({section.unit_id for section in deduped}),
        "sections_parsed": len(sections),
        "dropped_short": len(sections) - len(with_body),
        "dropped_near_duplicate": n_duplicates,
        "docs": len(deduped),
        "queries": len(queries),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tinylab-build-lumen", description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="directory of markdown files")
    parser.add_argument("--out", type=Path, required=True, help="where to write the evaluation set")
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[".ko", "-2.md", "_copy", "NEWER_", "FIRST_ORIGINAL_", "README_NEW"],
        help="substrings of relative paths to exclude",
    )
    args = parser.parse_args(argv)

    if not args.source.is_dir():
        print(f"error: {args.source} is not a directory")
        return 2

    stats = build(args.source, args.out, tuple(args.skip))
    for name, value in stats.items():
        print(f"  {name:<24} {value}")
    print(f"\nwrote {args.out}/corpus.jsonl and {args.out}/queries.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
