"""The evaluation loop. Frozen; a retriever is anything with index() and rank()."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from .metrics import aggregate, per_query_metrics
from .schema import Doc, Query, check_referential_integrity, load_docs, load_queries
from .splits import Split, split_of_query

DEFAULT_KS = (1, 5, 10, 20)
DEFAULT_TOP_K = 100


@runtime_checkable
class Retriever(Protocol):
    name: str

    def index(self, docs: list[Doc]) -> None: ...

    def rank(self, query_text: str, top_k: int) -> list[str]:
        """Return doc_ids best-first, at most top_k of them."""


@dataclass(slots=True)
class EvalResult:
    system: str
    split: str
    n_queries: int
    n_docs: int
    metrics: dict[str, float]
    per_query: dict[str, dict[str, float]] = field(repr=False, default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvalSet:
    docs: list[Doc]
    queries: list[Query]

    @classmethod
    def load(cls, directory: Path) -> EvalSet:
        docs = load_docs(directory / "corpus.jsonl")
        queries = load_queries(directory / "queries.jsonl")
        check_referential_integrity(docs, queries)
        for query in queries:
            split_of_query(query, docs)  # fail on a straddling query at load, not mid-run
        return cls(docs, queries)

    def queries_in(self, split: Split | None) -> list[Query]:
        if split is None:
            return list(self.queries)
        return [query for query in self.queries if split_of_query(query, self.docs) == split]


def evaluate(
    retriever: Retriever,
    eval_set: EvalSet,
    split: Split | None,
    ks: tuple[int, ...] = DEFAULT_KS,
    top_k: int = DEFAULT_TOP_K,
) -> EvalResult:
    """Score a retriever over the whole corpus, on the queries of one split."""
    if top_k < max(ks):
        raise ValueError(f"top_k={top_k} is below max(ks)={max(ks)}; recall would be truncated")

    queries = eval_set.queries_in(split)
    if not queries:
        raise ValueError(f"no queries in split {split!r}; nothing to score")

    retriever.index(eval_set.docs)

    per_query: dict[str, dict[str, float]] = {}
    for query in queries:
        ranking = retriever.rank(query.text, top_k)
        if len(ranking) != len(set(ranking)):
            raise ValueError(f"{retriever.name} returned duplicate doc_ids for {query.query_id}")
        per_query[query.query_id] = per_query_metrics(ranking, query, ks)

    return EvalResult(
        system=retriever.name,
        split=str(split) if split else "all",
        n_queries=len(queries),
        n_docs=len(eval_set.docs),
        metrics=aggregate(per_query),
        per_query=per_query,
    )
