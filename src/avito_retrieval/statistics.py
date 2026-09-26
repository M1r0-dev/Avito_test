"""Paired statistical inference for retrieval experiments.

Retrievers are evaluated on identical queries, so independent-sample tests are
inappropriate. All comparisons operate on per-query metric differences.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairedTestResult:
    """Mean paired effect, bootstrap interval and one-sided randomization p-value."""

    mean_delta: float
    ci_low: float
    ci_high: float
    p_value_greater: float
    n_queries: int


def per_query_recall(
    predictions: Mapping[str, Sequence[str]],
    relevant: Mapping[str, set[str]],
    *,
    k: int = 50,
    query_ids: Sequence[str] | None = None,
) -> dict[str, float]:
    """Return query-level recall contributions used by macro Recall@k."""
    ids = list(query_ids) if query_ids is not None else list(relevant)
    result: dict[str, float] = {}
    for query_id in ids:
        truth = relevant[query_id]
        if truth:
            result[query_id] = len(set(predictions.get(query_id, ())[:k]) & truth) / len(truth)
    return result


def paired_recall_test(
    candidate: Mapping[str, Sequence[str]],
    baseline: Mapping[str, Sequence[str]],
    relevant: Mapping[str, set[str]],
    *,
    k: int = 50,
    query_ids: Sequence[str] | None = None,
    n_resamples: int = 20_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> PairedTestResult:
    """Compare Recall@k with paired bootstrap CI and sign-randomization test.

    The one-sided null is that the candidate has no positive mean improvement.
    Sign randomization is exact under exchangeability; Monte Carlo is used for
    practical speed and includes the observed sample via the +1 correction.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    cand = per_query_recall(candidate, relevant, k=k, query_ids=query_ids)
    base = per_query_recall(baseline, relevant, k=k, query_ids=query_ids)
    ids = sorted(set(cand) & set(base))
    if not ids:
        raise ValueError("No common labeled queries")
    differences = np.asarray([cand[q] - base[q] for q in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)

    # Generate in chunks to avoid a n_resamples × n_queries allocation spike.
    bootstrap_means: list[np.ndarray] = []
    random_means: list[np.ndarray] = []
    chunk_size = 1_000
    for start in range(0, n_resamples, chunk_size):
        size = min(chunk_size, n_resamples - start)
        indices = rng.integers(0, len(differences), size=(size, len(differences)))
        bootstrap_means.append(differences[indices].mean(axis=1))
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, len(differences)))
        random_means.append((signs * differences).mean(axis=1))
    bootstrap = np.concatenate(bootstrap_means)
    randomized = np.concatenate(random_means)
    observed = float(differences.mean())
    p_value = float((1 + np.count_nonzero(randomized >= observed)) / (n_resamples + 1))
    alpha = 1 - confidence
    low, high = np.quantile(bootstrap, [alpha / 2, 1 - alpha / 2])
    return PairedTestResult(observed, float(low), float(high), p_value, len(ids))


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for filter compatibility proportions."""
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("Require 0 <= successes <= trials and trials > 0")
    # 1.95996 is the standard-normal 97.5th percentile for a 95% interval.
    if confidence != 0.95:
        raise ValueError("Only the pre-registered 95% interval is supported")
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    center = (proportion + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)) / denominator
    return float(center - margin), float(center + margin)
