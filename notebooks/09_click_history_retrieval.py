# %% [markdown]
# # Click-history retrieval for typo and paraphrase transfer
#
# BM25, dense и SPLADE ищут по тексту объявления. Здесь проверяется ортогональный
# сигнал: какие объявления выбирали по похожим поисковым формулировкам. Символьные
# n-граммы устойчивы к опечаткам, словоформам и коротким русским запросам.
#
# Защита от утечки и протокол:
#
# - из истории до построения индекса удаляются все строки с любой из пяти
#   validation query signatures;
# - corpus ограничивается `benchmark_items`, category и минимальный rating
#   применяются как hard filters;
# - параметры kNN и RRF выбираются только на прежнем dev-fold;
# - test используется один раз для выбранной конфигурации;
# - это шестой model-family look: primary `alpha=0.05/6`, CI=99.1667%;
# - инженерный выбор принимается только при положительном paired effect,
#   bootstrap CI выше нуля и sign-randomization `p < alpha`.

# %%
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import json
import math
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

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
PRIMARY_ALPHA = 0.05 / 6
PRIMARY_CONFIDENCE = 1 - PRIMARY_ALPHA
HISTORY_OUTPUT_K = 250


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).replace("ё", "е").lower().split())


def parse_rank(value: object) -> list[str]:
    return [] if value is None or pd.isna(value) or not str(value) else str(value).split()


def subset(mapping: dict, ids: set[str]) -> dict:
    return {query_id: mapping[query_id] for query_id in ids}


# %% [markdown]
# ## 1. Неизменный holdout и leakage-safe history
#
# Anti-join использует все пять query fields, а не один текст. Это удаляет
# собственные клики validation-запроса, но сохраняет допустимую историю похожих
# запросов и того же текста в другой локации/с другими фильтрами.

# %%
manifest = pd.read_parquet(ROOT / "artifacts/validation/manifest.parquet").copy()
labels = pd.read_parquet(ROOT / "artifacts/validation/labels.parquet")
train = pd.read_parquet(ROOT / "dataset/train.parquet")
items = pd.read_parquet(
    ROOT / "dataset/benchmark_items.parquet",
    columns=["item_id", "item_category_id", "item_location_id", "item_rating"],
)
benchmark = pd.read_parquet(ROOT / "dataset/benchmark_queries.parquet")
bm25 = pd.read_parquet(ROOT / "artifacts/rankings/bm25_rankings.parquet")
dense = pd.read_parquet(ROOT / "artifacts/dense_kaggle/dense_rankings.parquet")
rankings = bm25.merge(dense, on=["query_key", "split"], validate="one_to_one")

relevant = labels.groupby("eval_query_id").item_id.agg(set).to_dict()
manifest["fold"] = -1
splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
for fold, (_, indices) in enumerate(splitter.split(manifest, manifest.primary_stratum)):
    manifest.loc[indices, "fold"] = fold
dev_ids = set(manifest.loc[manifest.fold.eq(0), "eval_query_id"])
test_ids = set(manifest.loc[manifest.fold.eq(1), "eval_query_id"])

marked = train.merge(
    manifest[QUERY_COLUMNS].drop_duplicates().assign(_validation=1),
    on=QUERY_COLUMNS, how="left",
)
history = marked[marked._validation.isna()].drop(columns="_validation").copy()
history["item_id"] = history.item_id.astype(str)
valid_items = set(items.item_id.astype(str))
history = history[history.item_id.isin(valid_items)]
history["query_norm"] = history.search_query.map(clean)
history = history[history.query_norm.ne("")]
assert len(history.merge(
    manifest[QUERY_COLUMNS].drop_duplicates(), on=QUERY_COLUMNS, how="inner"
)) == 0
print({"history_rows": len(history), "unique_queries": history.query_norm.nunique()})


# %% [markdown]
# ## 2. Символьный индекс и click postings
#
# `char_wb` 3–5 grams выбран до просмотра метрик: он сохраняет границы слов и
# устойчив к частичным совпадениям. `min_df=2` удаляет единичный шум, лимит 200k
# признаков ограничивает RAM. Индекс возвращает 50 ближайших исторических
# формулировок; параметры отсечения выбираются далее только на dev.

# %%
history_queries = np.asarray(sorted(history.query_norm.unique()))
vectorizer = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=200_000,
    sublinear_tf=True, dtype=np.float32, norm="l2",
)
history_matrix = vectorizer.fit_transform(history_queries)
neighbors = NearestNeighbors(n_neighbors=50, metric="cosine", algorithm="brute", n_jobs=-1)
neighbors.fit(history_matrix)

click_counts = history.groupby(["query_norm", "item_id"]).size().rename("clicks").reset_index()
postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
for row in click_counts.itertuples(index=False):
    postings[str(row.query_norm)].append((str(row.item_id), int(row.clicks)))

item_meta = items.copy()
item_meta["item_id"] = item_meta.item_id.astype(str)
item_meta = item_meta.set_index("item_id")
item_category = item_meta.item_category_id.astype(int).to_dict()
item_location = item_meta.item_location_id.astype(int).to_dict()
item_rating = item_meta.item_rating.fillna(-1).astype(float).to_dict()

validation_queries = manifest.rename(columns={"eval_query_id": "query_key"})
benchmark_queries = benchmark.copy()
benchmark_queries["query_key"] = benchmark_queries.query_id.astype(str)
all_queries = pd.concat([
    validation_queries[QUERY_COLUMNS + ["query_key"]],
    benchmark_queries[QUERY_COLUMNS + ["query_key"]],
], ignore_index=True)
query_matrix = vectorizer.transform(all_queries.search_query.map(clean))
distances, neighbor_indices = neighbors.kneighbors(query_matrix, return_distance=True)
similarities = 1.0 - distances


def history_ranking(
    query_no: int,
    *,
    neighbor_k: int,
    min_similarity: float,
    location_boost: float,
) -> list[str]:
    query = all_queries.iloc[query_no]
    category = int(query.search_category)
    location = int(query.search_location_id)
    min_rating = requested_min_rating(query.search_infm_params_text)
    scores: dict[str, float] = defaultdict(float)
    best_similarity: dict[str, float] = defaultdict(float)
    for similarity, neighbor_row in zip(
        similarities[query_no, :neighbor_k], neighbor_indices[query_no, :neighbor_k], strict=True
    ):
        similarity = float(similarity)
        if similarity < min_similarity:
            continue
        historical_query = history_queries[int(neighbor_row)]
        for item_id, clicks in postings[historical_query]:
            if category and item_category[item_id] != category:
                continue
            rating = item_rating[item_id]
            if min_rating is not None and rating < min_rating:
                continue
            local = location_boost if item_location[item_id] == location else 1.0
            scores[item_id] += similarity**2 * math.log1p(clicks) * local
            best_similarity[item_id] = max(best_similarity[item_id], similarity)
    return sorted(scores, key=lambda x: (-scores[x], -best_similarity[x], x))[:HISTORY_OUTPUT_K]


def build_history_predictions(
    config: tuple[int, float, float], query_positions: range | None = None,
) -> dict[str, list[str]]:
    neighbor_k, threshold, location_boost = config
    positions = query_positions if query_positions is not None else range(len(all_queries))
    return {
        str(all_queries.iloc[query_no].query_key): history_ranking(
            query_no, neighbor_k=neighbor_k,
            min_similarity=threshold, location_boost=location_boost,
        )
        for query_no in positions
    }


# %% [markdown]
# ## 3. Dev-only selection и RRF
#
# Сначала выбирается сам history retriever по dev Recall@50. Затем для уже
# выбранного канала на dev настраиваются только два RRF параметра. Такой
# последовательный поиск проще интерпретировать, чем единая большая сетка.

# %%
history_grid = []
history_cache = {}
validation_positions = range(len(validation_queries))
for neighbor_k in [5, 10, 20, 50]:
    for threshold in [0.35, 0.50, 0.65]:
        for location_boost in [1.0, 1.5, 2.0]:
            key = (neighbor_k, threshold, location_boost)
            prediction = build_history_predictions(key, validation_positions)
            history_cache[key] = prediction
            history_grid.append({
                "neighbor_k": neighbor_k, "min_similarity": threshold,
                "location_boost": location_boost,
                "dev_recall": recall_at_k(subset(prediction, dev_ids), subset(relevant, dev_ids)),
                "dev_coverage": np.mean([bool(prediction[q]) for q in dev_ids]),
            })
history_grid = pd.DataFrame(history_grid).sort_values(
    ["dev_recall", "dev_coverage", "neighbor_k"], ascending=[False, False, True]
)
selected_history_key = (
    int(history_grid.iloc[0].neighbor_k),
    float(history_grid.iloc[0].min_similarity),
    float(history_grid.iloc[0].location_boost),
)
history_pred = history_cache[selected_history_key]
display(history_grid.head(12))
print("selected history:", selected_history_key)

validation_rankings = rankings[rankings.split.eq("validation")].set_index("query_key")
base_wide = {}
baseline = {}
for query_id, row in validation_rankings.iterrows():
    dense_rank = reciprocal_rank_fusion(
        [parse_rank(row.dense_global), parse_rank(row.dense_local)],
        weights=[1.0, 1.15], rrf_k=60, top_k=250,
    )
    base_wide[query_id] = reciprocal_rank_fusion(
        [parse_rank(row.bm25), dense_rank], weights=[1.0, 1.25], rrf_k=20, top_k=250,
    )
    baseline[query_id] = base_wide[query_id][:50]

rrf_grid, rrf_cache = [], {}
for rrf_k in [10, 20, 40, 60, 100]:
    for history_weight in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        prediction = {
            query_id: reciprocal_rank_fusion(
                [base_wide[query_id], history_pred[query_id]],
                weights=[1.0, history_weight], rrf_k=rrf_k, top_k=50,
            )
            for query_id in validation_rankings.index
        }
        rrf_cache[(rrf_k, history_weight)] = prediction
        rrf_grid.append({
            "rrf_k": rrf_k, "history_weight": history_weight,
            "dev_recall": recall_at_k(subset(prediction, dev_ids), subset(relevant, dev_ids)),
        })
rrf_grid = pd.DataFrame(rrf_grid).sort_values(
    ["dev_recall", "history_weight"], ascending=[False, True]
)
selected_rrf = (int(rrf_grid.iloc[0].rrf_k), float(rrf_grid.iloc[0].history_weight))
candidate = rrf_cache[selected_rrf]
display(rrf_grid.head(12))
print("selected RRF:", selected_rrf)


# %% [markdown]
# ## 4. Единственная test-оценка и candidate-pool ceiling
#
# Primary endpoint — прирост выбранного RRF над ранее зафиксированным
# BM25+dense. Дополнительно измеряется, приносит ли history новые positives в
# широкий pool; это определяет, стоит ли добавлять канал в следующий LTR.

# %%
primary = asdict(paired_recall_test(
    candidate, baseline, relevant, query_ids=sorted(test_ids),
    confidence=PRIMARY_CONFIDENCE,
))
history_secondary = asdict(paired_recall_test(
    history_pred, baseline, relevant, query_ids=sorted(test_ids),
))
baseline_test = recall_at_k(subset(baseline, test_ids), subset(relevant, test_ids))
candidate_test = recall_at_k(subset(candidate, test_ids), subset(relevant, test_ids))
accepted = bool(
    candidate_test > baseline_test and primary["ci_low"] > 0
    and primary["p_value_greater"] < PRIMARY_ALPHA
)

old_pool = {
    q: list(dict.fromkeys([
        *parse_rank(validation_rankings.loc[q].bm25)[:200], *base_wide[q][:200],
    ]))
    for q in validation_rankings.index
}
new_pool = {
    q: list(dict.fromkeys([*old_pool[q], *history_pred[q][:200]]))
    for q in validation_rankings.index
}
pool_test = asdict(paired_recall_test(
    new_pool, old_pool, relevant, query_ids=sorted(test_ids), k=600,
    confidence=PRIMARY_CONFIDENCE,
))
pool_metrics = {
    "old": recall_at_k(subset(old_pool, test_ids), subset(relevant, test_ids), k=400),
    "with_history": recall_at_k(subset(new_pool, test_ids), subset(relevant, test_ids), k=600),
}
display(pd.DataFrame([
    {"method": "BM25+dense", "test_recall": baseline_test},
    {"method": "history", "test_recall": recall_at_k(subset(history_pred, test_ids), subset(relevant, test_ids))},
    {"method": "BM25+dense+history", "test_recall": candidate_test},
]))
display(pd.DataFrame({"primary": primary, "history_secondary": history_secondary, "pool": pool_test}).T)
print({"accepted": accepted, "candidate_pool": pool_metrics})


# %% [markdown]
# ## 5. Экспорт канала
#
# Rankings сохраняются независимо от принятия standalone RRF: статистически
# значимый прирост pool ceiling может оправдать использование history только как
# feature/candidate source в LTR. Файл содержит validation и benchmark в одном
# контракте с другими retrieval stages.

# %%
selected_all = build_history_predictions(selected_history_key)
history_rows = []
validation_keys = set(validation_rankings.index)
for row in all_queries.itertuples(index=False):
    query_key = str(row.query_key)
    history_rows.append({
        "query_key": query_key,
        "split": "validation" if query_key in validation_keys else "benchmark",
        "history": " ".join(selected_all[query_key]),
    })
history_frame = pd.DataFrame(history_rows)
assert history_frame.query_key.is_unique
(ROOT / "artifacts/rankings").mkdir(parents=True, exist_ok=True)
history_frame.to_parquet(ROOT / "artifacts/rankings/history_rankings.parquet", index=False)

metrics = {
    "method": "char_wb_tfidf_click_history",
    "history_rows_after_holdout_exclusion": len(history),
    "unique_historical_queries": len(history_queries),
    "selected_history": {
        "neighbor_k": selected_history_key[0],
        "min_similarity": selected_history_key[1],
        "location_boost": selected_history_key[2],
    },
    "selected_rrf": {"rrf_k": selected_rrf[0], "history_weight": selected_rrf[1]},
    "history_grid": history_grid.to_dict("records"),
    "rrf_grid": rrf_grid.to_dict("records"),
    "baseline_test_recall": baseline_test,
    "candidate_test_recall": candidate_test,
    "primary_alpha": PRIMARY_ALPHA,
    "bootstrap_confidence": PRIMARY_CONFIDENCE,
    "primary_test": primary,
    "history_secondary_test": history_secondary,
    "accepted": accepted,
    "candidate_pool": pool_metrics,
    "candidate_pool_test": pool_test,
}
(ROOT / "reports/click_history_metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("saved history rankings and metrics")
