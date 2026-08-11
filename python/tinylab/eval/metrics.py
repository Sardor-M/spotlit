"""Per-query retrieval metrics. Frozen; nDCG@10 is primary per ADR-0002."""

from __future__ import annotations

from math import log2

from .schema import Query


def recall_at_k(ranking: list[str], query: Query, k: int) -> float:
    """Fraction of relevant docs in the top k; grades ignored."""
    relevant = {doc_id for doc_id, grade in query.relevant.items() if grade > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranking[:k])) / len(relevant)


def reciprocal_rank(ranking: list[str], query: Query) -> float:
    """1/rank of the first relevant hit, 0 if there is none."""
    relevant = {doc_id for doc_id, grade in query.relevant.items() if grade > 0}
    for position, doc_id in enumerate(ranking, 1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranking: list[str], query: Query, k: int) -> float:
    """Graded nDCG: gain 2**grade - 1, discount 1/log2(rank + 1)."""
    gains = [2 ** query.relevant.get(doc_id, 0) - 1 for doc_id in ranking[:k]]
    dcg = sum(gain / log2(rank + 1) for rank, gain in enumerate(gains, 1) if gain)

    ideal = sorted(query.relevant.values(), reverse=True)[:k]
    idcg = sum((2**grade - 1) / log2(rank + 1) for rank, grade in enumerate(ideal, 1) if grade)
    return dcg / idcg if idcg else 0.0


def per_query_metrics(ranking: list[str], query: Query, ks: tuple[int, ...]) -> dict[str, float]:
    scores: dict[str, float] = {"mrr": reciprocal_rank(ranking, query)}
    for k in ks:
        scores[f"recall@{k}"] = recall_at_k(ranking, query, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(ranking, query, k)
    return scores


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(per_query: dict[str, dict[str, float]]) -> dict[str, float]:
    if not per_query:
        return {}
    names = next(iter(per_query.values())).keys()
    return {name: mean([scores[name] for scores in per_query.values()]) for name in names}
