"""Filter-aware retrieval channels shared by evaluation and submission."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .filters import category_mask, location_mask, requested_min_rating
from .fusion import reciprocal_rank_fusion
from .text import query_text


def top_rows(scores: np.ndarray, allowed_mask: np.ndarray, top_k: int) -> np.ndarray:
    """Select top rows after a pre-filter without fully sorting the corpus."""
    filtered = np.where(allowed_mask, scores, -np.inf)
    k = min(top_k, int(np.isfinite(filtered).sum()))
    if k == 0:
        return np.array([], dtype=np.int64)
    candidate = np.argpartition(filtered, -k)[-k:]
    return candidate[np.argsort(filtered[candidate])[::-1]]


def filtered_ranking_from_scores(
    query: pd.Series,
    items: pd.DataFrame,
    scores: np.ndarray,
    *,
    retrieve_k: int = 250,
    output_k: int = 250,
    rrf_k: int = 60,
    extra_mask: np.ndarray | None = None,
) -> list[str]:
    """Fuse global/local channels from one precomputed score vector."""
    global_mask = allowed_item_mask(items, query, local=False)
    if extra_mask is not None:
        global_mask &= extra_mask
    global_rows = top_rows(scores, global_mask, retrieve_k)
    rankings = [items.iloc[global_rows].item_id.astype(str).tolist()]
    weights = [1.0]
    local_mask = allowed_item_mask(items, query, local=True)
    if extra_mask is not None:
        local_mask &= extra_mask
    if not bool(query.search_is_delivery_search) and local_mask.any():
        local_rows = top_rows(scores, local_mask, retrieve_k)
        rankings.append(items.iloc[local_rows].item_id.astype(str).tolist())
        weights.append(1.15)
    return reciprocal_rank_fusion(rankings, weights=weights, rrf_k=rrf_k, top_k=output_k)


def allowed_item_mask(items: pd.DataFrame, query: pd.Series, *, local: bool) -> np.ndarray:
    """Build safe hard filters; location is only hard inside a parallel channel."""
    mask = category_mask(items, int(query.search_category))
    if local and not bool(query.search_is_delivery_search):
        mask &= location_mask(items, int(query.search_location_id))
    min_rating = requested_min_rating(query.search_infm_params_text)
    if min_rating is not None:
        ratings = items.item_rating.fillna(-1).to_numpy()
        mask &= ratings >= min_rating
    return mask


def filtered_sparse_ranking(
    query: pd.Series,
    items: pd.DataFrame,
    search: Callable[..., tuple[np.ndarray, np.ndarray]],
    *,
    retrieve_k: int = 250,
    output_k: int = 250,
    rrf_k: int = 60,
) -> list[str]:
    """Fuse global and same-location sparse results at item level."""
    text = query_text(query)
    global_rows, _ = search(
        text,
        top_k=retrieve_k,
        allowed_mask=allowed_item_mask(items, query, local=False),
    )
    rankings = [items.iloc[global_rows].item_id.astype(str).tolist()]
    weights = [1.0]

    local_mask = allowed_item_mask(items, query, local=True)
    # Delivery searches are intentionally global. Also avoid an empty channel.
    if not bool(query.search_is_delivery_search) and local_mask.any():
        local_rows, _ = search(text, top_k=retrieve_k, allowed_mask=local_mask)
        rankings.append(items.iloc[local_rows].item_id.astype(str).tolist())
        weights.append(1.15)
    return reciprocal_rank_fusion(rankings, weights=weights, rrf_k=rrf_k, top_k=output_k)
