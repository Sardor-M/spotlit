"""Deterministic split assignment, keyed on the correlated unit. Frozen; see ADR-0002."""

from __future__ import annotations

import hashlib
from collections import Counter
from enum import StrEnum

from .schema import Doc, Query, SchemaError

HASH_SALT = "tinylab-split-v1"  # changing this reshuffles every unit and voids all history


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    SEALED = "sealed"


def bucket_of(unit_id: str) -> int:
    """Stable 0-99 bucket, independent of platform and corpus order."""
    digest = hashlib.sha256(f"{HASH_SALT}:{unit_id}".encode()).hexdigest()
    return int(digest[:8], 16) % 100


def split_of_unit(unit_id: str) -> Split:
    bucket = bucket_of(unit_id)
    if bucket < 10:
        return Split.VAL
    if bucket < 20:
        return Split.TEST
    if bucket < 30:
        return Split.SEALED
    return Split.TRAIN


def split_of_query(query: Query, docs: list[Doc]) -> Split:
    """Split of the units a query judges; raises if they disagree."""
    unit_by_doc = {doc.doc_id: doc.unit_id for doc in docs}
    splits = {split_of_unit(unit_by_doc[doc_id]) for doc_id in query.relevant}
    if len(splits) > 1:
        detail = ", ".join(
            f"{doc_id}->{split_of_unit(unit_by_doc[doc_id])}" for doc_id in sorted(query.relevant)
        )
        raise SchemaError(
            f"query {query.query_id!r} straddles splits ({detail}); "
            "judgements for one query must come from units in a single split"
        )
    return splits.pop()


def summarize(docs: list[Doc], queries: list[Query]) -> dict:
    units = {doc.unit_id for doc in docs}
    unit_counts = Counter(split_of_unit(unit) for unit in units)
    doc_counts = Counter(split_of_unit(doc.unit_id) for doc in docs)
    query_counts = Counter(split_of_query(query, docs) for query in queries)
    return {
        "units": {split.value: unit_counts.get(split, 0) for split in Split},
        "docs": {split.value: doc_counts.get(split, 0) for split in Split},
        "queries": {split.value: query_counts.get(split, 0) for split in Split},
    }
