"""Data contract for evaluation sets. Frozen; see README.md for the format."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class SchemaError(ValueError):
    """An evaluation set violated the data contract."""


@dataclass(frozen=True, slots=True)
class Doc:
    doc_id: str
    unit_id: str
    text: str


@dataclass(frozen=True, slots=True)
class Query:
    query_id: str
    text: str
    relevant: dict[str, int]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SchemaError(f"missing evaluation file: {path}")
    records = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SchemaError(f"{path}:{lineno}: {exc}") from exc
    return records


def load_docs(path: Path) -> list[Doc]:
    docs = []
    seen: set[str] = set()
    for raw in _read_jsonl(path):
        for field in ("doc_id", "unit_id", "text"):
            if field not in raw:
                raise SchemaError(f"{path}: doc missing {field!r}: {raw}")
        if raw["doc_id"] in seen:
            raise SchemaError(f"{path}: duplicate doc_id {raw['doc_id']!r}")
        seen.add(raw["doc_id"])
        docs.append(Doc(str(raw["doc_id"]), str(raw["unit_id"]), str(raw["text"])))
    if not docs:
        raise SchemaError(f"{path}: corpus is empty")
    return docs


def load_queries(path: Path) -> list[Query]:
    queries = []
    seen: set[str] = set()
    for raw in _read_jsonl(path):
        for field in ("query_id", "text", "relevant"):
            if field not in raw:
                raise SchemaError(f"{path}: query missing {field!r}: {raw}")
        if raw["query_id"] in seen:
            raise SchemaError(f"{path}: duplicate query_id {raw['query_id']!r}")
        seen.add(raw["query_id"])
        relevant = {str(k): int(v) for k, v in dict(raw["relevant"]).items()}
        if not relevant:
            raise SchemaError(f"{path}: query {raw['query_id']!r} has no relevant docs")
        if any(grade < 0 for grade in relevant.values()):
            raise SchemaError(f"{path}: query {raw['query_id']!r} has a negative grade")
        if all(grade == 0 for grade in relevant.values()):
            raise SchemaError(f"{path}: query {raw['query_id']!r} has no positive grade")
        queries.append(Query(str(raw["query_id"]), str(raw["text"]), relevant))
    if not queries:
        raise SchemaError(f"{path}: query set is empty")
    return queries


def check_referential_integrity(docs: list[Doc], queries: list[Query]) -> None:
    """Fail if a judgement points at a doc_id that is not in the corpus."""
    known = {doc.doc_id for doc in docs}
    for query in queries:
        missing = sorted(set(query.relevant) - known)
        if missing:
            raise SchemaError(
                f"query {query.query_id!r} judges doc_ids absent from the corpus: {missing}"
            )
