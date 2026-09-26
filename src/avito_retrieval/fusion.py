"""Rank fusion and item-level de-duplication."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence


def unique_top_k(item_ids: Iterable[str], k: int = 50) -> list[str]:
    """Preserve rank while removing repeated chunks of the same item."""
    seen: set[str] = set()
    result: list[str] = []
    for item_id in item_ids:
        if item_id not in seen:
            seen.add(item_id)
            result.append(item_id)
            if len(result) == k:
                break
    return result


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    weights: Sequence[float] | None = None,
    rrf_k: int = 60,
    top_k: int = 50,
) -> list[str]:
    """Fuse item rankings without requiring score calibration."""
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(rankings) != len(weights):
        raise ValueError("Each ranking must have exactly one weight")

    scores: dict[str, float] = defaultdict(float)
    best_rank: dict[str, int] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item_id in enumerate(unique_top_k(ranking, k=len(ranking)), start=1):
            scores[item_id] += weight / (rrf_k + rank)
            best_rank[item_id] = min(rank, best_rank.get(item_id, rank))
    ordered = sorted(scores, key=lambda item: (-scores[item], best_rank[item], item))
    return ordered[:top_k]

