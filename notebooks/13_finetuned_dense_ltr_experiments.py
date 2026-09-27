# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Replace zero-shot dense with fine-tuned dense inside LTR
#
# BM25+dense уже имеет test Recall@50 `0.8092`, но union их top-250 имеет
# Recall около `0.94`. Значит, основная потеря сейчас происходит не при поиске
# широкого пула, а при сжатии пула до 50 items. Этот notebook проверяет
# supervised learning-to-rank как более дешёвую альтернативу ColBERT. После
# принятого domain adaptation проверяем замену dense channel без увеличения
# числа retrieval sources или ширины pool.
#
# Защита от утечки:
#
# - все точные query signatures validation полностью исключаются из click-history;
# - исходный dev-fold дополнительно делится на fit/tune со стратификацией;
# - features и CatBoost family зафиксированы предыдущим LTR;
# - iterations нового LTR выбираются только на tune;
# - control строится из сохранённых zero-shot pair features и 195 iterations;
# - это десятый model-family look: `alpha=0.005`, CI=99.5%.

# %%
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import gc
import json
import math
import re
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from avito_retrieval.filters import requested_min_rating
from avito_retrieval.fusion import reciprocal_rank_fusion
from avito_retrieval.metrics import recall_at_k
from avito_retrieval.statistics import paired_recall_test

QUERY_COLUMNS = [
    "search_query", "search_location_id", "search_is_delivery_search",
    "search_infm_params_text", "search_category",
]
TOKEN_RE = re.compile(r"(?u)[\w]+(?:[-'][\w]+)*")
MISSING_RANK = 501
POOL_WIDTH = 200
PRIMARY_ALPHA = 0.05 / 10
PRIMARY_CONFIDENCE = 1 - PRIMARY_ALPHA
CONTROL_ITERATIONS = 195


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).replace("ё", "е").lower().split())


def tokens(value: object) -> frozenset[str]:
    return frozenset(TOKEN_RE.findall(clean(value)))


def parse_rank(value: object) -> list[str]:
    return [] if value is None or pd.isna(value) or not str(value) else str(value).split()


def rank_map(values: list[str]) -> dict[str, int]:
    return {item_id: rank for rank, item_id in enumerate(values, 1)}


def subset(mapping: dict, ids: set[str]) -> dict:
    return {query_id: mapping[query_id] for query_id in ids}


def top_from_scores(frame: pd.DataFrame, score_column: str, k: int = 50) -> dict[str, list[str]]:
    ordered = frame.sort_values(
        ["query_key", score_column, "baseline_rank", "item_id"],
        ascending=[True, False, True, True],
    )
    return ordered.groupby("query_key", sort=False).head(k).groupby("query_key").item_id.agg(list).to_dict()


def overlap_features(query_tokens: frozenset[str], item_tokens: frozenset[str]) -> tuple[int, float, float]:
    overlap = len(query_tokens & item_tokens)
    coverage = overlap / max(len(query_tokens), 1)
    jaccard = overlap / max(len(query_tokens | item_tokens), 1)
    return overlap, coverage, jaccard


def safe_nonnegative(value: object) -> float:
    return max(float(value), 0.0) if value is not None and pd.notna(value) else 0.0


# %% [markdown]
# ## 1. Данные, неизменный split и широкий candidate pool
#
# Candidate pool — union top-200 BM25, уже выбранного dense global/local и
# SPLADE q32/d192. SPLADE не принят как самостоятельный RRF-канал, но его
# добавление в широкий pool стоит проверить: оно увеличивает oracle recall и не
# заставляет ranker автоматически помещать sparse-кандидаты в top-50.
# Ширина 200 выбрана как минимальный практически безопасный бюджет по dev-oracle
# при локальном лимите памяти; test-oracle до фиксации модели не используется.

# %%
manifest = pd.read_parquet(ROOT / "artifacts/validation/manifest.parquet").copy()
labels = pd.read_parquet(ROOT / "artifacts/validation/labels.parquet")
benchmark = pd.read_parquet(ROOT / "dataset/benchmark_queries.parquet")
items = pd.read_parquet(ROOT / "dataset/benchmark_items.parquet")
train = pd.read_parquet(ROOT / "dataset/train.parquet")

bm25 = pd.read_parquet(ROOT / "artifacts/rankings/bm25_rankings.parquet")
fine = pd.read_parquet(ROOT / "artifacts/finetuned_dense_kaggle/finetuned_dense_rankings.parquet")
splade = pd.read_parquet(ROOT / "artifacts/splade_kaggle/splade_rankings.parquet")
rankings = bm25.merge(fine, on=["query_key", "split"], validate="one_to_one").merge(
    splade, on=["query_key", "split"], validate="one_to_one"
)

relevant = labels.groupby("eval_query_id").item_id.agg(set).to_dict()
manifest["fold"] = -1
outer = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
for fold, (_, indices) in enumerate(outer.split(manifest, manifest.primary_stratum)):
    manifest.loc[indices, "fold"] = fold
dev_ids = set(manifest.loc[manifest.fold.eq(0), "eval_query_id"])
test_ids = set(manifest.loc[manifest.fold.eq(1), "eval_query_id"])

dev_table = manifest[manifest.eval_query_id.isin(dev_ids)]
fit_values, tune_values = train_test_split(
    dev_table.eval_query_id,
    test_size=0.25,
    random_state=42,
    stratify=dev_table.primary_stratum,
)
fit_ids, tune_ids = set(fit_values), set(tune_values)
print({"fit": len(fit_ids), "tune": len(tune_ids), "test": len(test_ids)})


def channel_lists(row: pd.Series) -> dict[str, list[str]]:
    bm25_rank = parse_rank(row.bm25)
    dense_rank = reciprocal_rank_fusion(
        [parse_rank(row.finetuned_global), parse_rank(row.finetuned_local)],
        weights=[1.0, 1.5], top_k=250,
    )
    splade_rank = reciprocal_rank_fusion(
        [parse_rank(row.splade_q32_d192_global), parse_rank(row.splade_q32_d192_local)],
        weights=[1.0, 2.0], rrf_k=60, top_k=250,
    )
    baseline = reciprocal_rank_fusion(
        [bm25_rank, dense_rank], weights=[1.0, 2.0], rrf_k=20, top_k=250
    )
    return {"bm25": bm25_rank, "dense": dense_rank, "splade": splade_rank, "baseline": baseline}


validation_rankings = rankings[rankings.split.eq("validation")].set_index("query_key")
channel_cache = {query_id: channel_lists(row) for query_id, row in validation_rankings.iterrows()}
baseline_pred = {query_id: values["baseline"][:50] for query_id, values in channel_cache.items()}
union_bd = {
    query_id: list(dict.fromkeys([*values["bm25"][:POOL_WIDTH], *values["dense"][:POOL_WIDTH]]))
    for query_id, values in channel_cache.items()
}
union_all = {
    query_id: list(dict.fromkeys([*union_bd[query_id], *values["splade"][:POOL_WIDTH]]))
    for query_id, values in channel_cache.items()
}
print("dev oracle BM25+dense:", recall_at_k(subset(union_bd, dev_ids), subset(relevant, dev_ids), k=2 * POOL_WIDTH))
print("dev oracle +SPLADE:", recall_at_k(subset(union_all, dev_ids), subset(relevant, dev_ids), k=3 * POOL_WIDTH))

# %% [markdown]
# ## 2. Leakage-safe history и pair features
#
# История вычисляется после anti-join по всем пяти query-полям: ни один клик
# validation query signature не может стать признаком самого себя. Exact
# `search_query → item/microcategory` history остаётся допустимым production
# сигналом для повторяющихся формулировок. Три вложенных feature sets отделяют
# вклад rank channels, content/metadata и истории.

# %%
marked = train.merge(
    manifest[QUERY_COLUMNS].drop_duplicates().assign(_validation_query=1),
    on=QUERY_COLUMNS, how="left",
)
history_train = marked[marked._validation_query.isna()].drop(columns="_validation_query").copy()
history_train["query_norm"] = history_train.search_query.map(clean)
item_pop = history_train.item_id.astype(str).value_counts().to_dict()
query_item_pop = history_train.groupby(["query_norm", "item_id"]).size().to_dict()
query_microcat_pop = history_train.groupby(["query_norm", "item_microcat_id"]).size().to_dict()
assert len(history_train) + marked._validation_query.notna().sum() == len(train)

item_frame = items.copy()
item_frame["item_id"] = item_frame.item_id.astype(str)
item_frame = item_frame.set_index("item_id", drop=False)
item_token_cache = {
    item_id: (
        tokens(row.item_title_raw), tokens(row.item_infm_params_text),
        tokens(row.item_description_raw),
    )
    for item_id, row in item_frame.iterrows()
}


def build_features(query_frame: pd.DataFrame, ranking_frame: pd.DataFrame) -> pd.DataFrame:
    query_by_key = query_frame.set_index("query_key")
    records: list[dict[str, object]] = []
    for position, (query_id, rank_row) in enumerate(ranking_frame.iterrows(), 1):
        query = query_by_key.loc[query_id]
        channels = channel_lists(rank_row)
        maps = {name: rank_map(values) for name, values in channels.items()}
        candidate_ids = list(dict.fromkeys([
            *channels["bm25"][:POOL_WIDTH], *channels["dense"][:POOL_WIDTH],
            *channels["splade"][:POOL_WIDTH],
        ]))
        q_words = tokens(f"{query.search_query} {query.search_infm_params_text}")
        filter_words = tokens(query.search_infm_params_text)
        q_norm = clean(query.search_query)
        min_rating = requested_min_rating(query.search_infm_params_text)
        for item_id in candidate_ids:
            item = item_frame.loc[item_id]
            title_words, param_words, description_words = item_token_cache[item_id]
            title_overlap = overlap_features(q_words, title_words)
            param_overlap = overlap_features(q_words, param_words)
            desc_overlap = overlap_features(q_words, description_words)
            filter_overlap = overlap_features(filter_words, param_words)
            channel_ranks = {name: maps[name].get(item_id, MISSING_RANK) for name in maps}
            source_count = sum(channel_ranks[name] <= POOL_WIDTH for name in ("bm25", "dense", "splade"))
            rating = float(item.item_rating) if pd.notna(item.item_rating) else -1.0
            records.append({
                "query_key": str(query_id), "item_id": item_id,
                "label": int(item_id in relevant.get(str(query_id), set())),
                "bm25_rank": channel_ranks["bm25"], "dense_rank": channel_ranks["dense"],
                "splade_rank": channel_ranks["splade"], "baseline_rank": channel_ranks["baseline"],
                "rr_bm25": 1.0 / (20 + channel_ranks["bm25"]),
                "rr_dense": 1.25 / (20 + channel_ranks["dense"]),
                "rr_splade": 0.5 / (20 + channel_ranks["splade"]),
                "min_source_rank": min(channel_ranks["bm25"], channel_ranks["dense"], channel_ranks["splade"]),
                "source_count": source_count,
                "title_overlap": title_overlap[0], "title_coverage": title_overlap[1], "title_jaccard": title_overlap[2],
                "param_overlap": param_overlap[0], "param_coverage": param_overlap[1], "param_jaccard": param_overlap[2],
                "desc_overlap": desc_overlap[0], "desc_coverage": desc_overlap[1], "desc_jaccard": desc_overlap[2],
                "filter_param_overlap": filter_overlap[0], "filter_param_coverage": filter_overlap[1],
                "exact_query_in_title": int(bool(q_norm) and q_norm in clean(item.item_title_raw)),
                "exact_query_in_description": int(bool(q_norm) and q_norm in clean(item.item_description_raw)),
                "location_match": int(int(query.search_location_id) == int(item.item_location_id)),
                "category_match": int(int(query.search_category) in (0, int(item.item_category_id))),
                "rating": rating,
                "rating_margin": rating - min_rating if min_rating is not None else 0.0,
                "log_reviews": math.log1p(safe_nonnegative(item.item_rating_reviews_count)),
                "log_price": math.log1p(safe_nonnegative(item.item_price)),
                "phone_hidden": int(item.item_is_phone_hidden == 1),
                "message_forbidden": int(item.item_is_message_forbidden == 1),
                "title_tokens": len(title_words), "description_tokens": len(description_words),
                "log_item_clicks": math.log1p(item_pop.get(item_id, 0)),
                "log_query_item_clicks": math.log1p(query_item_pop.get((q_norm, item_id), 0)),
                "log_query_microcat_clicks": math.log1p(query_microcat_pop.get((q_norm, item.item_microcat_id), 0)),
            })
        if position % 250 == 0:
            print(f"features {position}/{len(ranking_frame)}")
    return pd.DataFrame.from_records(records)


validation_queries = manifest.rename(columns={"eval_query_id": "query_key"})
validation_features = build_features(validation_queries, validation_rankings)
for column in validation_features.select_dtypes(include=["float64"]).columns:
    validation_features[column] = validation_features[column].astype("float32")
for column in validation_features.select_dtypes(include=["int64"]).columns:
    validation_features[column] = pd.to_numeric(validation_features[column], downcast="integer")
print(validation_features.shape, "positives in pool=", int(validation_features.label.sum()))
control_validation_features = pd.read_parquet(
    ROOT / "artifacts/features/ltr_validation_features.parquet"
)
assert set(control_validation_features.query_key) == set(validation_features.query_key)
# The token-set cache dominates RAM and is not needed while CatBoost trains.
del item_token_cache
gc.collect()

# %% [markdown]
# ## 3. Dev-only iteration selection
#
# CatBoostRanker оптимизирует pairwise порядок внутри query group. Для обучения
# оставляем все positives и максимум 200 самых трудных negatives по минимальному
# рангу каналов: случайные далёкие negatives почти не учат границу top-50, но
# резко увеличивают время. Набор `history` уже принят предыдущим experiment и
# здесь не переоптимизируется. Tune выбирает только число iterations нового LTR.

# %%
rank_features = [
    "bm25_rank", "dense_rank", "splade_rank", "baseline_rank",
    "rr_bm25", "rr_dense", "rr_splade", "min_source_rank", "source_count",
]
content_features = rank_features + [
    "title_overlap", "title_coverage", "title_jaccard",
    "param_overlap", "param_coverage", "param_jaccard",
    "desc_overlap", "desc_coverage", "desc_jaccard",
    "filter_param_overlap", "filter_param_coverage",
    "exact_query_in_title", "exact_query_in_description",
    "location_match", "category_match", "rating", "rating_margin",
    "log_reviews", "log_price", "phone_hidden", "message_forbidden",
    "title_tokens", "description_tokens",
]
history_features = content_features + [
    "log_item_clicks", "log_query_item_clicks", "log_query_microcat_clicks",
]


def hard_negative_subset(frame: pd.DataFrame, ids: set[str], max_rows: int = 201) -> pd.DataFrame:
    selected = frame[frame.query_key.isin(ids)].sort_values(
        ["query_key", "label", "min_source_rank"], ascending=[True, False, True]
    )
    return selected.groupby("query_key", sort=False).head(max_rows).reset_index(drop=True)


def make_pool(frame: pd.DataFrame, features: list[str]) -> Pool:
    group_id = pd.factorize(frame.query_key, sort=False)[0].astype("int32")
    return Pool(frame[features], label=frame.label, group_id=group_id)


fit_frame = hard_negative_subset(validation_features, fit_ids)
tune_frame = validation_features[validation_features.query_key.isin(tune_ids)].copy()
tuner = CatBoostRanker(
    loss_function="YetiRankPairwise", eval_metric="NDCG:top=50",
    iterations=500, depth=7, learning_rate=0.05,
    random_seed=42, thread_count=-1, verbose=100,
    od_type="Iter", od_wait=60, allow_writing_files=False,
)
tuner.fit(make_pool(fit_frame, history_features), eval_set=make_pool(tune_frame, history_features))
selected_iterations = max(tuner.get_best_iteration() + 1, 1)
tune_frame["score"] = tuner.predict(tune_frame[history_features])
candidate_tune = top_from_scores(tune_frame, "score")

control_fit = hard_negative_subset(control_validation_features, fit_ids)
control_tune_frame = control_validation_features[
    control_validation_features.query_key.isin(tune_ids)
].copy()
control_tuner = CatBoostRanker(
    loss_function="YetiRankPairwise", iterations=CONTROL_ITERATIONS,
    depth=7, learning_rate=0.05, random_seed=42, thread_count=-1,
    verbose=100, allow_writing_files=False,
)
control_tuner.fit(make_pool(control_fit, history_features))
control_tune_frame["score"] = control_tuner.predict(control_tune_frame[history_features])
control_tune = top_from_scores(control_tune_frame, "score")
tune_comparison = asdict(paired_recall_test(
    candidate_tune, control_tune, relevant, query_ids=sorted(tune_ids),
))
print({
    "selected_iterations": selected_iterations,
    "candidate_tune": recall_at_k(candidate_tune, subset(relevant, tune_ids)),
    "control_tune": recall_at_k(control_tune, subset(relevant, tune_ids)),
})
display(pd.DataFrame({"fine_dense_LTR_vs_control_tune": tune_comparison}).T)

# %% [markdown]
# ## 4. Единственная primary-оценка на test
#
# После выбора iterations candidate и control переобучаются на одном dev. Test
# используется для одного paired сравнения. Принятие требует CI выше нуля,
# `p<0.005` и итоговый Recall@50 выше control.

# %%
dev_train = hard_negative_subset(validation_features, dev_ids)
candidate_model = CatBoostRanker(
    loss_function="YetiRankPairwise", iterations=selected_iterations,
    depth=7, learning_rate=0.05, random_seed=42, thread_count=-1,
    verbose=100, allow_writing_files=False,
)
candidate_model.fit(make_pool(dev_train, history_features))
test_frame = validation_features[validation_features.query_key.isin(test_ids)].copy()
test_frame["ltr_score"] = candidate_model.predict(test_frame[history_features])
candidate_test = top_from_scores(test_frame, "ltr_score")

control_dev = hard_negative_subset(control_validation_features, dev_ids)
control_model = CatBoostRanker(
    loss_function="YetiRankPairwise", iterations=CONTROL_ITERATIONS,
    depth=7, learning_rate=0.05, random_seed=42, thread_count=-1,
    verbose=100, allow_writing_files=False,
)
control_model.fit(make_pool(control_dev, history_features))
control_test_frame = control_validation_features[
    control_validation_features.query_key.isin(test_ids)
].copy()
control_test_frame["ltr_score"] = control_model.predict(control_test_frame[history_features])
control_test = top_from_scores(control_test_frame, "ltr_score")

primary = asdict(paired_recall_test(
    candidate_test, control_test, relevant, query_ids=sorted(test_ids),
    confidence=PRIMARY_CONFIDENCE,
))
baseline_test = recall_at_k(subset(baseline_pred, test_ids), subset(relevant, test_ids))
control_test_recall = recall_at_k(control_test, subset(relevant, test_ids))
candidate_test_recall = recall_at_k(candidate_test, subset(relevant, test_ids))
primary_confirmed = bool(
    primary["mean_delta"] > 0 and primary["ci_low"] > 0
    and primary["p_value_greater"] < PRIMARY_ALPHA
    and candidate_test_recall > control_test_recall
)
display(pd.DataFrame([
    {"method": "BM25+dense", "test_recall": baseline_test},
    {"method": "zero-shot dense LTR control", "test_recall": control_test_recall},
    {"method": "fine-tuned dense LTR", "test_recall": candidate_test_recall},
]))
display(pd.DataFrame({"fine_dense_LTR_vs_control": primary}).T)
print("distance to 0.9:", 0.9 - candidate_test_recall, "accepted:", primary_confirmed)

# %% [markdown]
# ## 5. Pool/segment diagnostics и benchmark answer
#
# Если LTR принят, production model обучается на всех validation queries тем же
# iteration budget. Benchmark признаки строятся идентично, без labels. Файл
# `answer_finetuned_dense_ltr.csv` сохраняется для аудита всегда; основной `answer.csv`
# заменяется только при выполнении primary-критерия.

# %%
pool_test = {
    "pool_width_per_channel": POOL_WIDTH,
    "BM25+dense_union": recall_at_k(subset(union_bd, test_ids), subset(relevant, test_ids), k=2 * POOL_WIDTH),
    "BM25+dense+SPLADE_union": recall_at_k(subset(union_all, test_ids), subset(relevant, test_ids), k=3 * POOL_WIDTH),
}
slice_rows = []
for column in ["has_filters", "local_pool_bucket", "query_frequency_bucket", "positive_bucket"]:
    for value, group in manifest[manifest.eval_query_id.isin(test_ids)].groupby(column):
        ids = set(group.eval_query_id)
        base = recall_at_k(subset(control_test, ids), subset(relevant, ids))
        new = recall_at_k(subset(candidate_test, ids), subset(relevant, ids))
        slice_rows.append({"slice": column, "value": str(value), "queries": len(ids),
                           "baseline": base, "ltr": new, "delta": new - base})
slices = pd.DataFrame(slice_rows)
display(slices)

all_train = hard_negative_subset(validation_features, set(relevant))
production_model = CatBoostRanker(
    loss_function="YetiRankPairwise", iterations=selected_iterations,
    depth=7, learning_rate=0.05, random_seed=42, thread_count=-1,
    verbose=100, allow_writing_files=False,
)
production_model.fit(make_pool(all_train, history_features))

benchmark_rankings = rankings[rankings.split.eq("benchmark")].set_index("query_key")
# Validation pair table is the dominant RAM consumer. It is no longer needed
# after the production model is fitted; keep only compact predictions/metrics.
del validation_features, control_validation_features, fit_frame, tune_frame
del control_fit, control_tune_frame, dev_train, test_frame, control_dev
del control_test_frame, all_train
del train, marked, history_train, candidate_model, control_model, tuner, control_tuner
gc.collect()
item_token_cache = {
    item_id: (
        tokens(row.item_title_raw), tokens(row.item_infm_params_text),
        tokens(row.item_description_raw),
    )
    for item_id, row in item_frame.iterrows()
}
benchmark_queries = benchmark.copy()
benchmark_queries["query_key"] = benchmark_queries.query_id.astype(str)
benchmark_features = build_features(benchmark_queries, benchmark_rankings)
for column in benchmark_features.select_dtypes(include=["float64"]).columns:
    benchmark_features[column] = benchmark_features[column].astype("float32")
for column in benchmark_features.select_dtypes(include=["int64"]).columns:
    benchmark_features[column] = pd.to_numeric(benchmark_features[column], downcast="integer")
benchmark_features["ltr_score"] = production_model.predict(benchmark_features[history_features])
benchmark_pred = top_from_scores(benchmark_features, "ltr_score")
answer = pd.DataFrame({
    "query_id": benchmark.query_id.astype(str),
    "answer": [" ".join(benchmark_pred[str(query_id)]) for query_id in benchmark.query_id],
})
valid_items = set(items.item_id.astype(str))
assert len(answer) == len(benchmark) and answer.query_id.is_unique
for value in answer.answer:
    ids = value.split()
    assert len(ids) == 50 and len(ids) == len(set(ids)) and set(ids) <= valid_items
    assert all(re.fullmatch(r"[0-9a-f]{16}", item_id) for item_id in ids)
answer.to_csv(ROOT / "answer_finetuned_dense_ltr.csv", index=False, encoding="utf-8")
if primary_confirmed:
    answer.to_csv(ROOT / "answer.csv", index=False, encoding="utf-8")

metrics = {
    "candidate_pool": pool_test,
    "selected_iterations": selected_iterations,
    "control_iterations": CONTROL_ITERATIONS,
    "tune_comparison": tune_comparison,
    "baseline_test_recall": baseline_test,
    "control_ltr_test_recall": control_test_recall,
    "candidate_ltr_test_recall": candidate_test_recall,
    "distance_to_0_9": 0.9 - candidate_test_recall,
    "primary_alpha": PRIMARY_ALPHA,
    "bootstrap_confidence": PRIMARY_CONFIDENCE,
    "primary_test": primary,
    "primary_confirmed": primary_confirmed,
    "slice_metrics": slices.to_dict("records"),
}
(ROOT / "reports/finetuned_dense_ltr_metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
)
display(answer.head())
print("answer_finetuned_dense_ltr.csv validated; promoted to answer.csv:", primary_confirmed)
