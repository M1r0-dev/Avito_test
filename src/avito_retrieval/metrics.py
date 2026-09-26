"""Offline candidate-generation metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def recall_at_k(
    predictions: Mapping[str, Sequence[str]],
    relevant: Mapping[str, set[str]],
    k: int = 50,
) -> float:
    """Compute macro Recall@k exactly as defined by the benchmark."""
    values: list[float] = []
    for query_id, truth in relevant.items():
        if not truth:
            continue
        retrieved = set(predictions.get(query_id, ())[:k])
        values.append(len(retrieved & truth) / len(truth))
    if not values:
        raise ValueError("No non-empty relevance sets were supplied")
    return sum(values) / len(values)


def hit_rate_at_k(
    predictions: Mapping[str, Sequence[str]],
    relevant: Mapping[str, set[str]],
    k: int = 50,
) -> float:
    """Share of queries with at least one relevant item in top-k."""
    hits = [bool(set(predictions.get(qid, ())[:k]) & truth) for qid, truth in relevant.items()]
    if not hits:
        raise ValueError("No queries were supplied")
    return sum(hits) / len(hits)

