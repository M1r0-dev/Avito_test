"""PU-bagged candidate selector over BM25 + zero-shot + LoRA v2 (public attempt 6).

Online path of the final candidate generator:

    channel rankings -> union of top-100 per channel -> RRF top-200 pool ->
    notebook-19 features (system C2) -> three PU-bagged CatBoost models ->
    mean reciprocal rank -> top-50

The research notebooks (19 builds the features, 24 trains the PU bags) keep the
same logic inline next to the experiments. This module is the inference path
for `scripts/generate_answer.py` and the latency benchmark; the script checks
that it reproduces the submitted answer.csv bit for bit.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker

from .fusion import reciprocal_rank_fusion
from .learned_fusion import ItemSide, _overlap, clean, parse_rank, tokens

POOL_DEPTH = 100      # per-channel depth of the candidate union (notebook 17)
TOP_POOL = 200        # RRF top-200 kept for the selector (notebook 24)
MISSING_RANK = 501
FUSED_RRF_K = 20
# Global/local weights of the dense channels, selected on dev (notebooks 05, 19).
ZERO_LOCAL_WEIGHT = 1.15
V2_LOCAL_WEIGHT = 2.0
RR_OFFSET = 20.0      # ensemble score 1 / (20 + rank), notebook 24
DENSE = ("zero", "v2")
SOURCES = ("bm25", *DENSE)
RANK_LISTS = ("bm25", "bm25_plain", "fused",
              *(f"{name}{suffix}" for name in DENSE for suffix in ("", "_global", "_local")))
ITEM_PRIORS = ("rating", "log_reviews", "log_price", "phone_hidden", "message_forbidden")
FEATURES = [*RANK_LISTS, "min_rank", "source_count", "query_tokens", "location_match",
            "exact_in_title", "exact_in_desc", "title_tokens", "desc_tokens",
            "title_overlap", "title_coverage", "title_jaccard", "params_overlap", "params_coverage",
            "params_jaccard", "desc_overlap", "desc_coverage", "desc_jaccard",
            "filter_overlap", "filter_coverage", *ITEM_PRIORS]


def channel_lists(bm25: Sequence[str], bm25_plain: Sequence[str],
                  zero_global: Sequence[str], zero_local: Sequence[str],
                  v2_global: Sequence[str], v2_local: Sequence[str]) -> dict[str, list[str]]:
    """Retrieval sub-channels of one query plus their dense and system RRF lists."""
    lists = {"bm25": list(bm25), "bm25_plain": list(bm25_plain),
             "zero_global": list(zero_global), "zero_local": list(zero_local),
             "v2_global": list(v2_global), "v2_local": list(v2_local)}
    lists["zero"] = reciprocal_rank_fusion([lists["zero_global"], lists["zero_local"]],
                                           weights=[1.0, ZERO_LOCAL_WEIGHT], rrf_k=60, top_k=250)
    lists["v2"] = reciprocal_rank_fusion([lists["v2_global"], lists["v2_local"]],
                                         weights=[1.0, V2_LOCAL_WEIGHT], rrf_k=60, top_k=250)
    lists["fused"] = reciprocal_rank_fusion([lists[name] for name in SOURCES],
                                            weights=[1.0] * len(SOURCES), rrf_k=FUSED_RRF_K, top_k=250)
    return lists


def channel_lists_from_row(row: object) -> dict[str, list[str]]:
    """Same lists from a merged row of the saved Kaggle/BM25 ranking parquet files."""
    return channel_lists(parse_rank(row.bm25), parse_rank(row.bm25_plain),
                         parse_rank(row.dense_global), parse_rank(row.dense_local),
                         parse_rank(row.finetuned_v2_global), parse_rank(row.finetuned_v2_local))


def pool_items(lists: dict[str, list[str]]) -> list[str]:
    return list(dict.fromkeys(item for name in SOURCES for item in lists[name][:POOL_DEPTH]))


def query_features(query: object, query_key: str, lists: dict[str, list[str]],
                   items: ItemSide) -> pd.DataFrame:
    """Feature rows of the RRF top-200 pool of one query, ordered by `fused`."""
    ranks = {name: {item: rank for rank, item in enumerate(lists[name], 1)} for name in RANK_LISTS}
    q_tokens = tokens(f"{query.search_query} {query.search_infm_params_text}")
    f_tokens = tokens(query.search_infm_params_text)
    q_text = clean(query.search_query)
    location = int(query.search_location_id)
    rows = []
    for item in pool_items(lists):
        fused = ranks["fused"].get(item, MISSING_RANK)
        if fused > TOP_POOL:
            continue
        cached = items.cache[item]
        row = {name: ranks[name].get(item, MISSING_RANK) for name in RANK_LISTS}
        row.update({
            "query_key": query_key, "item_id": item,
            "min_rank": min(row[name] for name in SOURCES),
            "source_count": sum(row[name] <= POOL_DEPTH for name in SOURCES),
            "query_tokens": len(q_tokens),
            "location_match": int(cached["location"] == location),
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
    frame = pd.DataFrame.from_records(rows)
    return frame.sort_values("fused", kind="stable").reset_index(drop=True)


def load_channels(root: Path, split: str) -> dict[str, dict[str, list[str]]]:
    """Channel lists of every query of `split` from the saved ranking artifacts."""
    rankings = (
        pd.read_parquet(root / "artifacts/rankings/bm25_rankings.parquet")
        .merge(pd.read_parquet(root / "artifacts/dense_kaggle/dense_rankings.parquet"),
               on=["query_key", "split"], validate="one_to_one")
        .merge(pd.read_parquet(root / "artifacts/finetuned_v2_kaggle/finetuned_v2_dense_rankings.parquet"),
               on=["query_key", "split"], validate="one_to_one")
    )
    rankings = rankings[rankings.split.eq(split)]
    return {str(row.query_key): channel_lists_from_row(row) for row in rankings.itertuples(index=False)}


def build_features(queries: pd.DataFrame, key_column: str,
                   channels: dict[str, dict[str, list[str]]], items: pd.DataFrame) -> pd.DataFrame:
    """Selector features of all `queries`, sorted by (query_key, fused) as in notebook 24."""
    keys = queries[key_column].astype(str)
    needed = {item for key in keys for item in pool_items(channels[key])}
    item_side = ItemSide.build(items[items.item_id.astype(str).isin(needed)])
    frame = pd.concat([
        query_features(query, key, channels[key], item_side)
        for key, query in zip(keys, queries.itertuples(index=False))
    ], ignore_index=True)
    return frame.sort_values(["query_key", "fused"], kind="stable").reset_index(drop=True)


def ensemble_top50(frame: pd.DataFrame, models: Sequence[CatBoostRanker]) -> dict[str, list[str]]:
    """Mean reciprocal rank of the PU-bag models; ties broken by `fused` (notebook 24)."""
    work = frame[["query_key", "item_id", "fused"]].copy()
    features = frame[FEATURES]
    scores = []
    for model in models:
        work["raw"] = model.predict(features)
        ranks = work.groupby("query_key", sort=False).raw.rank(method="first", ascending=False)
        scores.append(1.0 / (RR_OFFSET + ranks.to_numpy()))
    work["score"] = np.mean(scores, axis=0)
    ordered = work.sort_values(["query_key", "score", "fused"], ascending=[True, False, True], kind="stable")
    return {query_id: values[:50] for query_id, values in
            ordered.groupby("query_key", sort=False).item_id.agg(list).items()}


# --- training (notebook 24) --------------------------------------------------
PU_BUDGET = 48            # unlabeled per query, selected on OOF dev in notebook 24
PU_SEEDS = (41, 42, 43)
HARD_FUSED, HARD_CHANNEL = 75, 40
TREES, DEPTH, LEARNING_RATE = 300, 6, 0.05


def pu_sample(frame: pd.DataFrame, negative_budget: int, seed: int) -> pd.DataFrame:
    """All positives + `negative_budget` unlabeled per query, 2/3 of them hard.

    Verbatim port of notebook 24 `pu_sample_best`: the RNG call order and the
    row order of the result are part of the reproducible model.
    """
    rng = np.random.default_rng(seed)
    pieces = []
    hard_budget = int(round(negative_budget * 2 / 3))
    for _, group in frame.groupby("query_key", sort=False):
        positive = group[group.label.eq(1)]
        negative = group[group.label.eq(0)]
        min_channel_rank = negative[list(SOURCES)].min(axis=1)
        hard = negative[negative.fused.le(HARD_FUSED) | min_channel_rank.le(HARD_CHANNEL)]
        other = negative[~negative.index.isin(hard.index)]

        take_hard = min(hard_budget, len(hard))
        hard_idx = rng.choice(hard.index.to_numpy(), size=take_hard, replace=False)
        remaining = negative_budget - take_hard
        take_other = min(remaining, len(other))
        other_idx = rng.choice(other.index.to_numpy(), size=take_other, replace=False)
        selected = set(hard_idx.tolist()) | set(other_idx.tolist())

        missing = min(negative_budget - len(selected), len(negative) - len(selected))
        if missing > 0:
            available = negative.index[~negative.index.isin(selected)].to_numpy()
            selected.update(rng.choice(available, size=missing, replace=False).tolist())
        pieces.append(pd.concat([positive, negative.loc[sorted(selected)]], axis=0))
    return pd.concat(pieces, ignore_index=True).sort_values("query_key", kind="stable").reset_index(drop=True)


def train_pu_models(frame: pd.DataFrame, seeds: Sequence[int] = PU_SEEDS,
                    negative_budget: int = PU_BUDGET) -> list[CatBoostRanker]:
    """One YetiRankPairwise model per PU bag; `frame` sorted by (query_key, fused)."""
    from catboost import Pool

    models = []
    for seed in seeds:
        bag = pu_sample(frame, negative_budget, seed)
        group_id = pd.factorize(bag.query_key, sort=False)[0].astype("int32")
        model = CatBoostRanker(loss_function="YetiRankPairwise", iterations=TREES, depth=DEPTH,
                               learning_rate=LEARNING_RATE, random_seed=seed, thread_count=-1,
                               verbose=False, allow_writing_files=False)
        model.fit(Pool(bag[FEATURES], label=bag.label, group_id=group_id))
        models.append(model)
    return models
