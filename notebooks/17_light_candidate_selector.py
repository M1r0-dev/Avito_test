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
# # Лёгкий learned fusion retrieval-каналов (замена RRF)
#
# **Рамка.** Это кандидатогенерация, а не ранжирование: вернуть 50
# объявлений с максимальным Recall@50 для последующего ranker'а. Модель ниже
# заменяет RRF как способ объединить уже посчитанные каналы; оптимизируется
# только принадлежность к 50, порядок внутри них метрике безразличен. Она не
# читает пару запрос–объявление нейросетью: признаки — ранги каналов и
# счётчики совпадающих токенов, поэтому стоит миллисекунды.
#
# **Latency.** Попытка 2 сама уже не укладывается в guardrail (p95 811.6 ms
# против 500 ms, `reports/LATENCY.md`), поэтому добавка селектора измеряется
# отдельно (раздел 5) и будет учитываться вместе с ускорением retrieval.
#
# **Почему.** Notebook 15: 75.8% промахов попытки 2 уже лежат в пуле
# BM25 + zero-shot dense + fine-tuned dense (медианный лучший ранг 67). RRF
# агрегирует только ранги; модель, видящая ранги всех под-каналов и дешёвые
# лексические совпадения, может лучше решить, какие 50 оставить.
#
# **Чем отличается от отклонённого LTR (notebook 13, public 0.698).** Там
# решающими были click-history признаки (`log_query_item_clicks`,
# `log_item_clicks`), которые на 63% незнакомых benchmark-запросов равны нулю,
# а модель училась на head-heavy holdout. Здесь признаки не зависят от истории
# запроса вовсе: только ранги каналов и текстовые совпадения, которые
# вычисляются одинаково для любого, в том числе нового, запроса.
#
# **Протокол (зафиксирован до обучения):**
#
# - пул: объединение top-100 трёх каналов попытки 2 (как `build_rerank_pool.py`);
# - feature sets: `A` — ранги + текст + локация; `B` — `A` + item priors
#   (рейтинг, отзывы, цена, флаги контактов), которые могут переносить
#   популярность продавца;
# - режимы: `selector` — top-50 по score модели; `blend` — RRF(селектор,
#   попытка 2) с равными весами как страховка от сдвига;
# - iterations `∈ {100, 300}`; итого 2 × 2 × 2 = 8 конфигураций;
# - обучение только на dev с 2-fold cross-fitting: выбор конфигурации по
#   out-of-fold Recall@50 на **dev-tail**, tie-break — OOF dev;
# - финальная модель переобучается на всём dev; одна проверка против
#   попытки 2 на **test-tail** (primary) и **test** (secondary), Bonferroni
#   `alpha = 0.05 / 8`;
# - latency селектора (признаки + predict) измеряется отдельно, batch=1.

# %%
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import math
import re
import sys
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from sklearn.model_selection import StratifiedKFold

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from avito_retrieval.fusion import reciprocal_rank_fusion
from avito_retrieval.metrics import recall_at_k
from avito_retrieval.statistics import paired_recall_test, per_query_recall

RANDOM_SEED = 42
TAIL_MAX_FREQUENCY = 1
ATTEMPT_2_WEIGHTS = (1.0, 0.75, 1.25)
RRF_K = 20
POOL_DEPTH = 100
MISSING_RANK = 501
ITERATIONS = (100, 300)
TOKEN_RE = re.compile(r"(?u)[\w]+(?:[-'][\w]+)*")


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


def select(mapping: dict, ids: set[str]) -> dict:
    return {query_id: mapping[query_id] for query_id in ids}


def safe_nonnegative(value: object) -> float:
    return max(float(value), 0.0) if value is not None and pd.notna(value) else 0.0


# %% [markdown]
# ## 1. Holdout, split и каналы попытки 2
#
# Внешний dev/test split тот же, что в notebooks 05–16 (`seed=42`). Внутри dev
# второй StratifiedKFold задаёт два fold'а для cross-fitting; test не
# участвует ни в обучении, ни в выборе.

# %%
manifest = pd.read_parquet(ROOT / "artifacts/validation/manifest.parquet").copy()
labels = pd.read_parquet(ROOT / "artifacts/validation/labels.parquet")
benchmark_queries = pd.read_parquet(ROOT / "dataset/benchmark_queries.parquet")
relevant = labels.groupby("eval_query_id").item_id.agg(set).to_dict()

manifest["fold"] = -1
outer = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_SEED)
for fold, (_, indices) in enumerate(outer.split(manifest, manifest.primary_stratum)):
    manifest.loc[indices, "fold"] = fold
dev_ids = set(manifest.loc[manifest.fold.eq(0), "eval_query_id"])
test_ids = set(manifest.loc[manifest.fold.eq(1), "eval_query_id"])
tail_ids = set(manifest.loc[manifest.query_frequency.le(TAIL_MAX_FREQUENCY), "eval_query_id"])

dev_frame = manifest[manifest.fold.eq(0)].reset_index(drop=True)
inner = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_SEED)
cross_folds = [
    (set(dev_frame.loc[fit, "eval_query_id"]), set(dev_frame.loc[held, "eval_query_id"]))
    for fit, held in inner.split(dev_frame, dev_frame.primary_stratum)
]

rankings = (
    pd.read_parquet(ROOT / "artifacts/rankings/bm25_rankings.parquet")
    .merge(pd.read_parquet(ROOT / "artifacts/dense_kaggle/dense_rankings.parquet"),
           on=["query_key", "split"], validate="one_to_one")
    .merge(pd.read_parquet(
        ROOT / "artifacts/finetuned_dense_kaggle/finetuned_dense_rankings.parquet"),
        on=["query_key", "split"], validate="one_to_one")
)
queries = pd.concat([
    manifest.assign(query_key=manifest.eval_query_id, split="validation"),
    benchmark_queries.assign(query_key=benchmark_queries.query_id, split="benchmark"),
], ignore_index=True)
queries = queries.set_index(["split", "query_key"])


def channel_lists(row: object) -> dict[str, list[str]]:
    """All sub-channels of attempt 2; each is already computed online."""
    lists = {
        "bm25": parse_rank(row.bm25),
        "bm25_plain": parse_rank(row.bm25_plain),
        "zero_global": parse_rank(row.dense_global),
        "zero_local": parse_rank(row.dense_local),
        "fine_global": parse_rank(row.finetuned_global),
        "fine_local": parse_rank(row.finetuned_local),
    }
    lists["zero"] = reciprocal_rank_fusion(
        [lists["zero_global"], lists["zero_local"]], weights=[1.0, 1.15], rrf_k=60, top_k=250)
    lists["fine"] = reciprocal_rank_fusion(
        [lists["fine_global"], lists["fine_local"]], weights=[1.0, 1.5], rrf_k=60, top_k=250)
    lists["attempt_2"] = reciprocal_rank_fusion(
        [lists["bm25"], lists["zero"], lists["fine"]],
        weights=list(ATTEMPT_2_WEIGHTS), rrf_k=RRF_K, top_k=250)
    return lists


channels = {(row.split, str(row.query_key)): channel_lists(row) for row in rankings.itertuples(index=False)}
attempt_2 = {query_id: lists["attempt_2"][:50]
             for (split, query_id), lists in channels.items() if split == "validation"}

# %% [markdown]
# ## 2. Признаки
#
# Всё, что ниже, вычисляется online за доли миллисекунды на кандидата: ранги
# уже есть после retrieval, токены объявлений считаются offline и хранятся
# рядом с индексом, для запроса токенизируется одна строка.

# %%
items = pd.read_parquet(ROOT / "dataset/benchmark_items.parquet", columns=[
    "item_id", "item_title_raw", "item_infm_params_text", "item_description_raw",
    "item_location_id", "item_category_id", "item_rating", "item_rating_reviews_count",
    "item_price", "item_is_phone_hidden", "item_is_message_forbidden",
])
items["item_id"] = items.item_id.astype(str)
pool_items = {
    item for lists in channels.values()
    for name in ("bm25", "zero", "fine") for item in lists[name][:POOL_DEPTH]
}
items = items[items.item_id.isin(pool_items)].set_index("item_id")
# Offline item side: token sets and normalized title/description strings.
item_cache = {
    item_id: {
        "title": tokens(row.item_title_raw), "params": tokens(row.item_infm_params_text),
        "desc": tokens(row.item_description_raw),
        "title_text": clean(row.item_title_raw), "desc_text": clean(row.item_description_raw),
        "location": int(row.item_location_id), "category": int(row.item_category_id),
        "rating": float(row.item_rating) if pd.notna(row.item_rating) else -1.0,
        "log_reviews": math.log1p(safe_nonnegative(row.item_rating_reviews_count)),
        "log_price": math.log1p(safe_nonnegative(row.item_price)),
        "phone_hidden": int(row.item_is_phone_hidden == 1),
        "message_forbidden": int(row.item_is_message_forbidden == 1),
    }
    for item_id, row in items.iterrows()
}
RANK_CHANNELS = ("bm25", "bm25_plain", "zero_global", "zero_local", "fine_global",
                 "fine_local", "zero", "fine", "attempt_2")


def overlap(query_tokens: frozenset[str], item_tokens: frozenset[str]) -> tuple[int, float, float]:
    shared = len(query_tokens & item_tokens)
    return shared, shared / max(len(query_tokens), 1), shared / max(len(query_tokens | item_tokens), 1)


def query_features(key: tuple[str, str]) -> pd.DataFrame:
    """Feature rows for every pool candidate of one query (online path)."""
    query = queries.loc[key]
    lists = channels[key]
    ranks = {name: {item: rank for rank, item in enumerate(lists[name], 1)} for name in RANK_CHANNELS}
    pool = list(dict.fromkeys(
        item for name in ("bm25", "zero", "fine") for item in lists[name][:POOL_DEPTH]))
    q_tokens = tokens(f"{query.search_query} {query.search_infm_params_text}")
    f_tokens = tokens(query.search_infm_params_text)
    q_text = clean(query.search_query)
    rows = []
    for item in pool:
        cached = item_cache[item]
        row = {name: ranks[name].get(item, MISSING_RANK) for name in RANK_CHANNELS}
        row.update({
            "query_key": key[1], "item_id": item,
            "min_rank": min(row["bm25"], row["zero"], row["fine"]),
            "source_count": sum(row[name] <= POOL_DEPTH for name in ("bm25", "zero", "fine")),
            "query_tokens": len(q_tokens),
            "location_match": int(cached["location"] == int(query.search_location_id)),
            "exact_in_title": int(bool(q_text) and q_text in cached["title_text"]),
            "exact_in_desc": int(bool(q_text) and q_text in cached["desc_text"]),
            "title_tokens": len(cached["title"]), "desc_tokens": len(cached["desc"]),
            **{key_name: cached[key_name] for key_name in
               ("rating", "log_reviews", "log_price", "phone_hidden", "message_forbidden")},
        })
        for field in ("title", "params", "desc"):
            row[f"{field}_overlap"], row[f"{field}_coverage"], row[f"{field}_jaccard"] = overlap(
                q_tokens, cached[field])
        row["filter_overlap"], row["filter_coverage"], _ = overlap(f_tokens, cached["params"])
        rows.append(row)
    return pd.DataFrame.from_records(rows)


FEATURES_A = [*RANK_CHANNELS, "min_rank", "source_count", "query_tokens", "location_match",
              "exact_in_title", "exact_in_desc", "title_tokens", "desc_tokens",
              "title_overlap", "title_coverage", "title_jaccard",
              "params_overlap", "params_coverage", "params_jaccard",
              "desc_overlap", "desc_coverage", "desc_jaccard",
              "filter_overlap", "filter_coverage"]
FEATURES_B = FEATURES_A + ["rating", "log_reviews", "log_price", "phone_hidden", "message_forbidden"]
FEATURE_SETS = {"A": FEATURES_A, "B": FEATURES_B}

validation_features = pd.concat(
    [query_features(("validation", query_id)) for query_id in manifest.eval_query_id],
    ignore_index=True)
validation_features["label"] = [
    int(item in relevant[query_id])
    for query_id, item in zip(validation_features.query_key, validation_features.item_id)
]
print(validation_features.shape, "positives in pool:", int(validation_features.label.sum()),
      "of", sum(len(v) for v in relevant.values()))

# %% [markdown]
# ## 3. Cross-fitted выбор конфигурации на dev
#
# `YetiRankPairwise` — лучший objective в notebook 12; depth и learning rate те
# же, что там. Кандидаты сортируются по score, ties — по рангу попытки 2.

# %%
def make_pool(frame: pd.DataFrame, features: list[str]) -> Pool:
    group_id = pd.factorize(frame.query_key, sort=False)[0].astype("int32")
    return Pool(frame[features], label=frame.label, group_id=group_id)


def train(ids: set[str], features: list[str], iterations: int) -> CatBoostRanker:
    frame = validation_features[validation_features.query_key.isin(ids)]
    model = CatBoostRanker(
        loss_function="YetiRankPairwise", iterations=iterations, depth=6,
        learning_rate=0.05, random_seed=RANDOM_SEED, thread_count=-1,
        verbose=False, allow_writing_files=False)
    model.fit(make_pool(frame, features))
    return model


def top50(frame: pd.DataFrame, scores: np.ndarray, mode: str) -> dict[str, list[str]]:
    ordered = frame.assign(score=scores).sort_values(
        ["query_key", "score", "attempt_2"], ascending=[True, False, True])
    ranked = ordered.groupby("query_key", sort=False).item_id.agg(list).to_dict()
    if mode == "selector":
        return {query_id: values[:50] for query_id, values in ranked.items()}
    return {  # blend: equal-weight RRF with the submitted attempt-2 ranking
        query_id: reciprocal_rank_fusion(
            [values, channels[(split, query_id)]["attempt_2"]],
            weights=[1.0, 1.0], rrf_k=RRF_K, top_k=50)
        for query_id, values in ranked.items()
        for split in ["validation" if query_id in relevant else "benchmark"]
    }


CONFIGS = [(fs, it, mode) for fs in FEATURE_SETS for it in ITERATIONS for mode in ("selector", "blend")]
config_name = lambda c: f"{c[0]}_it{c[1]}_{c[2]}"
oof = {config_name(c): {} for c in CONFIGS}
for fit_ids, held_ids in cross_folds:
    held = validation_features[validation_features.query_key.isin(held_ids)]
    for feature_set in FEATURE_SETS:
        # Gradient boosting is sequential: the first 100 trees of a 300-tree
        # model are exactly the 100-iteration model, so one fit serves both.
        model = train(fit_ids, FEATURE_SETS[feature_set], max(ITERATIONS))
        for iterations in ITERATIONS:
            scores = model.predict(held[FEATURE_SETS[feature_set]], ntree_end=iterations)
            for mode in ("selector", "blend"):
                oof[config_name((feature_set, iterations, mode))].update(top50(held, scores, mode))

rows = [{"method": "attempt_2",
         "oof_dev_tail": recall_at_k(select(attempt_2, dev_ids & tail_ids), select(relevant, dev_ids & tail_ids)),
         "oof_dev": recall_at_k(select(attempt_2, dev_ids), select(relevant, dev_ids))}]
for name, predictions in oof.items():
    rows.append({"method": name,
                 "oof_dev_tail": recall_at_k(select(predictions, dev_ids & tail_ids), select(relevant, dev_ids & tail_ids)),
                 "oof_dev": recall_at_k(select(predictions, dev_ids), select(relevant, dev_ids))})
dev_comparison = pd.DataFrame(rows)
display(dev_comparison.round(5))
challengers = dev_comparison[dev_comparison.method.ne("attempt_2")].sort_values(
    ["oof_dev_tail", "oof_dev"], ascending=[False, False], kind="stable")
selected_name = str(challengers.iloc[0].method)
selected = next(c for c in CONFIGS if config_name(c) == selected_name)
print("selected on OOF dev-tail:", selected_name)

# %% [markdown]
# ## 4. Финальная модель на dev и единственная проверка на test

# %%
final_model = train(dev_ids, FEATURE_SETS[selected[0]], selected[1])
test_frame = validation_features[validation_features.query_key.isin(test_ids)]
test_predictions = top50(test_frame, final_model.predict(test_frame[FEATURE_SETS[selected[0]]]), selected[2])

alpha = 0.05 / len(CONFIGS)
tests, directions = {}, {}
for segment, ids in {"test_tail": test_ids & tail_ids, "test": test_ids}.items():
    tests[segment] = paired_recall_test(
        test_predictions, attempt_2, relevant, query_ids=sorted(ids),
        confidence=1 - alpha, n_resamples=20_000, seed=RANDOM_SEED)
    new = per_query_recall(test_predictions, relevant, query_ids=sorted(ids))
    old = per_query_recall(attempt_2, relevant, query_ids=sorted(ids))
    delta = np.asarray([new[q] - old[q] for q in sorted(ids)])
    directions[segment] = {"wins": int((delta > 0).sum()), "ties": int((delta == 0).sum()),
                           "losses": int((delta < 0).sum())}
display(pd.DataFrame({segment: asdict(result) for segment, result in tests.items()}).T)
print(directions)
primary_confirmed = bool(tests["test_tail"].p_value_greater < alpha and tests["test_tail"].mean_delta > 0)
secondary_confirmed = bool(tests["test"].p_value_greater < alpha and tests["test"].mean_delta > 0)
print({"alpha": alpha, "primary_confirmed": primary_confirmed, "secondary_confirmed": secondary_confirmed})
# Ranking losses need the training pool to compute LossFunctionChange importance.
dev_pool = make_pool(validation_features[validation_features.query_key.isin(dev_ids)], FEATURE_SETS[selected[0]])
importance = pd.Series(final_model.get_feature_importance(data=dev_pool),
                       index=FEATURE_SETS[selected[0]]).sort_values(ascending=False)
display(importance.head(15).round(2))

# %% [markdown]
# ## 5. Latency селектора, batch=1
#
# Замеряется online-добавка к retrieval: построение признаков для пула одного
# запроса и `predict`. 25 warm-up + 500 test-запросов, как в guardrail-протоколе.

# %%
latency_ids = sorted(test_ids)[:525]
timings = []
for position, query_id in enumerate(latency_ids):
    started = time.perf_counter_ns()
    frame = query_features(("validation", query_id))
    final_model.predict(frame[FEATURE_SETS[selected[0]]])
    if position >= 25:
        timings.append((time.perf_counter_ns() - started) / 1e6)
selector_latency = {name: float(np.quantile(timings, q)) for name, q in
                    (("p50_ms", 0.5), ("p95_ms", 0.95), ("p99_ms", 0.99))}
selector_latency["mean_ms"] = float(np.mean(timings))
display(selector_latency)

# %% [markdown]
# ## 6. Benchmark-кандидат
#
# Файл пишется всегда как `answer_light_selector.csv`; `answer.csv` меняется
# только при подтверждённом primary или secondary endpoint.

# %%
benchmark_features = pd.concat(
    [query_features(("benchmark", str(query_id))) for query_id in benchmark_queries.query_id],
    ignore_index=True)
benchmark_predictions = top50(
    benchmark_features, final_model.predict(benchmark_features[FEATURE_SETS[selected[0]]]), selected[2])
corpus_ids = set(pd.read_parquet(ROOT / "dataset/benchmark_items.parquet", columns=["item_id"]).item_id)
answer = pd.DataFrame({
    "query_id": benchmark_queries.query_id.astype(str),
    "answer": [" ".join(benchmark_predictions[str(q)]) for q in benchmark_queries.query_id],
})
assert len(answer) == len(benchmark_queries) == answer.query_id.nunique()
for item_string in answer.answer:
    item_ids = item_string.split()
    assert 1 <= len(item_ids) <= 50 and len(item_ids) == len(set(item_ids))
    assert all(re.fullmatch(r"[0-9a-f]{16}", item_id) for item_id in item_ids)
    assert set(item_ids) <= corpus_ids
answer.to_csv(ROOT / "answer_light_selector.csv", index=False)
if primary_confirmed or secondary_confirmed:
    answer.to_csv(ROOT / "answer.csv", index=False)
submitted = pd.read_csv(ROOT / "submissions/attempt_2_robust_rrf.csv", dtype=str).set_index("query_id").answer
overlap_with_attempt_2 = float(np.mean([
    len(set(benchmark_predictions[q]) & set(submitted[q].split())) / 50 for q in submitted.index]))
print(f"benchmark overlap with attempt 2: {overlap_with_attempt_2:.3f}")

report = {
    "configs": [config_name(c) for c in CONFIGS],
    "pool_depth": POOL_DEPTH,
    "feature_sets": FEATURE_SETS,
    "dev_comparison": dev_comparison.to_dict("records"),
    "selected_on_oof_dev_tail": selected_name,
    "test_recall": {segment: recall_at_k(select(test_predictions, ids), select(relevant, ids))
                    for segment, ids in {"test_tail": test_ids & tail_ids, "test": test_ids}.items()},
    "bonferroni_alpha": alpha,
    "selected_vs_attempt_2": {segment: asdict(result) for segment, result in tests.items()},
    "directions": directions,
    "primary_confirmed": primary_confirmed,
    "secondary_confirmed": secondary_confirmed,
    "feature_importance": importance.round(4).to_dict(),
    "selector_latency_cpu_ms": selector_latency,
    "benchmark_overlap_with_attempt_2": overlap_with_attempt_2,
}
(ROOT / "reports/light_selector_metrics.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print("saved reports/light_selector_metrics.json")
