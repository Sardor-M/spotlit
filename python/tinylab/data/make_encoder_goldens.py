"""Generate (or check) the cross-language encoder parity vectors. See ADR-0005."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..encoder import SPEC_V1, EncoderSpec, features

DEFAULT_OUT = Path(__file__).resolve().parents[3] / "tests" / "encoder_goldens.txt"

CASES: list[tuple[str, str]] = [
    ("empty", ""),
    ("single-space", " "),
    ("whitespace-only", " \t\n  "),
    ("one-char", "a"),
    ("two-chars", "ab"),
    ("exactly-three", "abc"),
    ("shorter-than-largest-ngram", "abcd"),
    ("ascii-sentence", "the quick brown fox jumps over the lazy dog"),
    ("mixed-case", "HeLLo WoRLD"),
    ("punctuation-heavy", "foo, bar; baz! (qux) -- quux?"),
    ("digits", "version 2.4.1 build 9000"),
    ("underscores-and-dashes", "snake_case kebab-case camelCase"),
    ("all-non-alphanumeric", "!!! ??? ### ..."),
    ("collapsing-whitespace", "a   \t\n  b"),
    ("leading-trailing-space", "   padded   "),
    ("accented-latin", "café naïve résumé Ångström"),
    ("turkish-dotted-i", "İstanbul TITLE Iıİi"),
    ("german-sharp-s", "STRASSE straße Straße"),
    ("cjk", "한국어 문서 검색 시스템"),
    ("japanese", "日本語のテキスト検索"),
    ("emoji", "search 🔍 the docs 🚀 now"),
    ("emoji-zwj", "family 👨‍👩‍👧‍👦 sequence"),
    ("rtl", "بحث في الوثائق"),
    ("mixed-scripts", "search 검색 検索 поиск"),
    ("markdown-residue", "**bold** `code` [link](http://x.y) > quote"),
    ("real-passage", (
        "Wrap many inserts in a single transaction. Committing per row forces a durable "
        "write each time and is the usual reason a bulk load crawls."
    )),
    ("repeated-token", "test " * 40),
    ("truncation-boundary", "a" * 512),
    ("truncation-just-over", "a" * 513),
    ("truncation-splits-utf8", "a" * 510 + "한국"),
    ("long-text", "lorem ipsum dolor sit amet consectetur adipiscing elit " * 30),
]


def render(name: str, text: str, spec: EncoderSpec) -> str:
    payload = text.encode("utf-8").hex() or "-"
    pairs = features(text, spec)
    body = " ".join(f"{bucket}:{count}" for bucket, count in pairs)
    return f"case {name} {payload} {len(pairs)}{' ' + body if body else ''}"


def build(spec: EncoderSpec) -> str:
    lines = [f"version {spec.version}", f"spec {spec.describe()}"]
    lines.extend(render(name, text, spec) for name, text in CASES)
    return "\n".join(lines) + "\n"


def check(path: Path, spec: EncoderSpec) -> int:
    """Verify the committed goldens still match what this encoder produces."""
    if not path.exists():
        print(f"error: {path} does not exist; run without --check first", file=sys.stderr)
        return 2

    expected = path.read_text(encoding="utf-8")
    actual = build(spec)
    if expected == actual:
        print(f"goldens match ({len(CASES)} cases)")
        return 0

    for lineno, (want, got) in enumerate(zip(expected.splitlines(), actual.splitlines()), 1):
        if want != got:
            print(f"mismatch at line {lineno}", file=sys.stderr)
            print(f"  golden:  {want[:120]}", file=sys.stderr)
            print(f"  current: {got[:120]}", file=sys.stderr)
            break
    print("\nThe encoder changed. Either revert, or regenerate and retrain.", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tinylab-encoder-goldens", description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="compare instead of writing")
    args = parser.parse_args(argv)

    if args.check:
        return check(args.out, SPEC_V1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build(SPEC_V1), encoding="utf-8")
    print(f"wrote {args.out} ({len(CASES)} cases, {SPEC_V1.describe()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
