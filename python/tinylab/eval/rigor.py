"""Keep/discard gate: paired bootstrap over queries. Frozen; see ADR-0002."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .harness import EvalResult
from .metrics import mean

BOOTSTRAP_RESAMPLES = 20_000
REQUIRED_P_BETTER = 0.95
MIN_SEEDS_STOCHASTIC = 3


@dataclass(slots=True)
class Comparison:
    metric: str
    incumbent: str
    candidate: str
    n_queries: int
    n_seeds: tuple[int, int]
    incumbent_mean: float
    candidate_mean: float
    mean_delta: float
    ci_low: float
    ci_high: float
    p_better: float
    min_delta: float
    verdict: str
    reason: str


def _averaged_per_query(results: list[EvalResult], metric: str) -> dict[str, float]:
    """Mean of one metric per query, averaged across seed runs."""
    if not results:
        raise ValueError("no results given")
    query_ids = set(results[0].per_query)
    for result in results[1:]:
        if set(result.per_query) != query_ids:
            raise ValueError(f"{result.system}: seed runs disagree on the query set")
    averaged = {}
    for query_id in query_ids:
        values = []
        for result in results:
            if metric not in result.per_query[query_id]:
                raise ValueError(f"{result.system}: metric {metric!r} not recorded")
            values.append(result.per_query[query_id][metric])
        averaged[query_id] = mean(values)
    return averaged


def compare(
    incumbent: list[EvalResult],
    candidate: list[EvalResult],
    metric: str = "ndcg@10",
    min_delta: float = 0.0,
    stochastic: bool = False,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> Comparison:
    """Paired bootstrap on the difference of means; returns KEEP, DISCARD or INSUFFICIENT."""
    old = _averaged_per_query(incumbent, metric)
    new = _averaged_per_query(candidate, metric)

    shared = sorted(set(old) & set(new))
    if not shared:
        raise ValueError("the two systems share no queries; they are not comparable")
    if len(shared) != len(old) or len(shared) != len(new):
        raise ValueError(
            f"query sets differ ({len(old)} vs {len(new)}, {len(shared)} shared); "
            "a paired comparison requires the identical query set"
        )

    deltas = [new[query_id] - old[query_id] for query_id in shared]
    observed = mean(deltas)

    rng = random.Random(seed)
    n = len(shared)
    resampled_means = []
    wins = 0
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += deltas[rng.randrange(n)]
        resample_mean = total / n
        resampled_means.append(resample_mean)
        if resample_mean > 0.0:
            wins += 1

    resampled_means.sort()
    p_better = wins / resamples
    ci_low = resampled_means[int(0.025 * resamples)]
    ci_high = resampled_means[min(int(0.975 * resamples), resamples - 1)]

    if stochastic and len(candidate) < MIN_SEEDS_STOCHASTIC:
        verdict = "INSUFFICIENT"
        reason = (
            f"stochastic system ran {len(candidate)} seed(s); "
            f"ADR-0002 requires {MIN_SEEDS_STOCHASTIC}"
        )
    elif p_better < REQUIRED_P_BETTER:
        verdict = "DISCARD"
        reason = f"P(better)={p_better:.3f} < {REQUIRED_P_BETTER}"
    elif observed < min_delta:
        verdict = "DISCARD"
        reason = f"delta={observed:+.4f} below the {min_delta:+.4f} minimum effect size"
    else:
        verdict = "KEEP"
        reason = f"P(better)={p_better:.3f}, delta={observed:+.4f}"

    return Comparison(
        metric=metric,
        incumbent=incumbent[0].system,
        candidate=candidate[0].system,
        n_queries=n,
        n_seeds=(len(incumbent), len(candidate)),
        incumbent_mean=mean(list(old.values())),
        candidate_mean=mean(list(new.values())),
        mean_delta=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        p_better=p_better,
        min_delta=min_delta,
        verdict=verdict,
        reason=reason,
    )
