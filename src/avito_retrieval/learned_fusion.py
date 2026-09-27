"""Learned fusion of retrieval channels (notebook 17, public attempt 3).

The model decides which 50 items of the channel pool are returned; it is part
of candidate generation, not a re-ranker. Features are channel ranks, token
overlaps between the query and the item, location match and item priors.
This module is the inference path used by `scripts/generate_answer.py`; the
research notebook keeps the same logic inline next to the experiment. Both
must produce identical features, which the script verifies through the sha256
of the submitted answer.csv.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .fusion import reciprocal_rank_fusion

TOKEN_RE = re.compile(r"(?u)[\w]+(?:[-'][\w]+)*")
ATTEMPT_2_WEIGHTS = (1.0, 0.75, 1.25)
RRF_K = 20
POOL_DEPTH = 100
MISSING_RANK = 501
RANK_CHANNELS = ("bm25", "bm25_plain", "zero_global", "zero_local", "fine_global",
                 "fine_local", "zero", "fine", "attempt_2")
ITEM_PRIORS = ("rating", "log_reviews", "log_price", "phone_hidden", "message_forbidden")
FEATURES_A = [*RANK_CHANNELS, "min_rank", "source_count", "query_tokens", "location_match",
              "exact_in_title", "exact_in_desc", "title_tokens", "desc_tokens",
              "title_overlap", "title_coverage", "title_jaccard",
              "params_overlap", "params_coverage", "params_jaccard",
              "desc_overlap", "desc_coverage", "desc_jaccard",
              "filter_overlap", "filter_coverage"]
FEATURES_B = FEATURES_A + list(ITEM_PRIORS)
ITEM_COLUMNS = ["item_id", "item_title_raw", "item_infm_params_text", "item_description_raw",
                "item_location_id", "item_rating", "item_rating_reviews_count", "item_price",
                "item_is_phone_hidden", "item_is_message_forbidden"]


def clean(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return " ".join(str(value).replace("ё", "е").lower().split())


def tokens(value: object) -> frozenset[str]:
    return frozenset(TOKEN_RE.findall(clean(value)))


def parse_rank(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)) or not str(value):
        return []
    return str(value).split()


def _nonnegative(value: object) -> float:
    return max(float(value), 0.0) if value is not None and pd.notna(value) else 0.0


def channel_lists(row: object) -> dict[str, list[str]]:
    """All sub-channels of attempt 2 for one ranking row (BM25 + two dense models)."""
    lists = {
        "bm25": parse_rank(row.bm25),
        "bm25_plain": parse_rank(row.bm25_plain),
        "zero_global": parse_rank(row.dense_global),
        "zero_local": parse_rank(row.dense_local),
        "fine_global": parse_rank(row.finetuned_global),
        "fine_local": parse_rank(row.finetuned_local),
    }
    # Dense global/local weights were selected on dev in notebooks 05 and 08.
    lists["zero"] = reciprocal_rank_fusion(
        [lists["zero_global"], lists["zero_local"]], weights=[1.0, 1.15], rrf_k=60, top_k=250)
    lists["fine"] = reciprocal_rank_fusion(
        [lists["fine_global"], lists["fine_local"]], weights=[1.0, 1.5], rrf_k=60, top_k=250)
    lists["attempt_2"] = reciprocal_rank_fusion(
        [lists["bm25"], lists["zero"], lists["fine"]],
        weights=list(ATTEMPT_2_WEIGHTS), rrf_k=RRF_K, top_k=250)
    return lists


def pool_of(lists: dict[str, list[str]]) -> list[str]:
    return list(dict.fromkeys(
        item for name in ("bm25", "zero", "fine") for item in lists[name][:POOL_DEPTH]))


@dataclass(frozen=True)
class ItemSide:
    """Offline item features: token sets, normalized strings, priors."""

    cache: dict[str, dict[str, object]]

    @classmethod
    def build(cls, items: pd.DataFrame) -> "ItemSide":
        cache = {}
        for item_id, row in items.set_index(items.item_id.astype(str)).iterrows():
            cache[item_id] = {
                "title": tokens(row.item_title_raw), "params": tokens(row.item_infm_params_text),
                "desc": tokens(row.item_description_raw),
                "title_text": clean(row.item_title_raw), "desc_text": clean(row.item_description_raw),
                "location": int(row.item_location_id),
                "rating": float(row.item_rating) if pd.notna(row.item_rating) else -1.0,
                "log_reviews": math.log1p(_nonnegative(row.item_rating_reviews_count)),
                "log_price": math.log1p(_nonnegative(row.item_price)),
                "phone_hidden": int(row.item_is_phone_hidden == 1),
                "message_forbidden": int(row.item_is_message_forbidden == 1),
            }
        return cls(cache)


def _overlap(query_tokens: frozenset[str], item_tokens: frozenset[str]) -> tuple[int, float, float]:
    shared = len(query_tokens & item_tokens)
    return shared, shared / max(len(query_tokens), 1), shared / max(len(query_tokens | item_tokens), 1)


def query_features(query: object, query_key: str, lists: dict[str, list[str]],
                   items: ItemSide) -> pd.DataFrame:
    """Feature rows for every pool candidate of one query (online path)."""
    ranks = {name: {item: rank for rank, item in enumerate(lists[name], 1)} for name in RANK_CHANNELS}
    q_tokens = tokens(f"{query.search_query} {query.search_infm_params_text}")
    f_tokens = tokens(query.search_infm_params_text)
    q_text = clean(query.search_query)
    rows = []
    for item in pool_of(lists):
        cached = items.cache[item]
        row = {name: ranks[name].get(item, MISSING_RANK) for name in RANK_CHANNELS}
        row.update({
            "query_key": query_key, "item_id": item,
            "min_rank": min(row["bm25"], row["zero"], row["fine"]),
            "source_count": sum(row[name] <= POOL_DEPTH for name in ("bm25", "zero", "fine")),
            "query_tokens": len(q_tokens),
            "location_match": int(cached["location"] == int(query.search_location_id)),
            "exact_in_title": int(bool(q_text) and q_text in cached["title_text"]),
            "exact_in_desc": int(bool(q_text) and q_text in cached["desc_text"]),
            "title_tokens": len(cached["title"]), "desc_tokens": len(cached["desc"]),
            **{name: cached[name] for name in ITEM_PRIORS},
        })
        for field in ("title", "params", "desc"):
            row[f"{field}_overlap"], row[f"{field}_coverage"], row[f"{field}_jaccard"] = _overlap(
                q_tokens, cached[field])
        row["filter_overlap"], row["filter_coverage"], _ = _overlap(f_tokens, cached["params"])
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def top50(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, list[str]]:
    """Top-50 by model score; ties broken by the attempt-2 RRF rank (as in notebook 17)."""
    ordered = frame.assign(score=scores).sort_values(
        ["query_key", "score", "attempt_2"], ascending=[True, False, True])
    ranked = ordered.groupby("query_key", sort=False).item_id.agg(list).to_dict()
    return {query_id: values[:50] for query_id, values in ranked.items()}
