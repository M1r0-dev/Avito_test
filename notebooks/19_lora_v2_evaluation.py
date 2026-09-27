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
# # LoRA v2: оценка канала и системы кандидатогенерации
#
# LoRA v2 (stages 10A/10B) обучен на 457 439 парах вместо 17 033, с маской
# ложных негативов, batch 32/GPU и LoRA r=32 на всех dense-слоях. Notebook 18
# показал, что v1 не даёт доказанного прироста на хвосте, где 72% benchmark.
#
# **Протокол зафиксирован до получения весов v2.**
#
# *Уровень канала (описательный, отдельное семейство тестов).* v2 против v1,
# dense-only, global/local RRF; local weight v2 выбирается на dev из
# `{1.0, 1.15, 1.5, 2.0}` (для v1 он был выбран так же в notebook 08).
# Recall@50 и покрытие @250 на test, test-tail, test-head; paired bootstrap
# 95% CI и one-sided randomization.
#
# *Уровень системы (решение).* Learned fusion notebook 17 с теми же
# гиперпараметрами (feature set B, 300 деревьев, selector) — они не
# перевыбираются, чтобы не раздувать множественность. Меняется только набор
# dense-каналов:
#
# - `C1` — BM25 + zero-shot + v1: кандидат 3, baseline;
# - `C2` — BM25 + zero-shot + v2: v2 вместо v1;
# - `C3` — BM25 + zero-shot + v1 + v2;
# - `C4` — BM25 + v2: один query encoder вместо двух, около −350 ms p95.
#
# Выбор среди C2–C4 по out-of-fold Recall@50 на dev-tail (2-fold
# cross-fitting на dev, как в notebook 17), tie-break — OOF dev, затем меньше
# encoder'ов. Одна проверка выбранного против C1 на test-tail (primary) и test
# (secondary), Bonferroni `alpha = 0.05 / 3`. Для прозрачности test-дельты
# всех трёх вариантов тоже печатаются, но решения по ним не принимаются.

# %%
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import math
import re
import sys

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
POOL_DEPTH = 100
MISSING_RANK = 501
TREES = 300  # notebook 17 selection, not re-tuned
V2_LOCAL_WEIGHTS = (1.0, 1.15, 1.5, 2.0)
TOKEN_RE = re.compile(r"(?u)[\w]+(?:[-'][\w]+)*")
V2_DIR = ROOT / "artifacts/finetuned_v2_kaggle"


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
# ## 1. Holdout, split и rankings v1/v2

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
    .merge(pd.read_parquet(ROOT / "artifacts/finetuned_dense_kaggle/finetuned_dense_rankings.parquet"),
           on=["query_key", "split"], validate="one_to_one")
    .merge(pd.read_parquet(V2_DIR / "finetuned_v2_dense_rankings.parquet"),
           on=["query_key", "split"], validate="one_to_one")
)
display(json.loads((V2_DIR / "finetuned_v2_dense_run.json").read_text()))

# %% [markdown]
# ## 2. Уровень канала: v2 против v1
#
# Сначала local weight v2 выбирается на dev (как у v1 в notebook 08), затем
# описательные paired-сравнения на test.

# %%
validation_rows = rankings[rankings.split.eq("validation")]
v1_dense = {
    row.query_key: reciprocal_rank_fusion(
        [parse_rank(row.finetuned_global), parse_rank(row.finetuned_local)],
        weights=[1.0, 1.5], rrf_k=60, top_k=250)
    for row in validation_rows.itertuples(index=False)
}


def v2_dense(weight: float, frame: pd.DataFrame) -> dict[str, list[str]]:
    return {
        row.query_key: reciprocal_rank_fusion(
            [parse_rank(row.finetuned_v2_global), parse_rank(row.finetuned_v2_local)],
            weights=[1.0, weight], rrf_k=60, top_k=250)
        for row in frame.itertuples(index=False)
    }


weight_scores = {
    weight: recall_at_k(select(v2_dense(weight, validation_rows), dev_ids), select(relevant, dev_ids))
    for weight in V2_LOCAL_WEIGHTS
}
V2_LOCAL_WEIGHT = max(V2_LOCAL_WEIGHTS, key=lambda weight: (weight_scores[weight], -weight))
print("v2 local weight selected on dev:", V2_LOCAL_WEIGHT, weight_scores)
v2_validation = v2_dense(V2_LOCAL_WEIGHT, validation_rows)

channel_rows = []
head_ids = test_ids - tail_ids
for segment, ids in {"test": test_ids, "test_tail": test_ids & tail_ids, "test_head": head_ids}.items():
    for k in (50, 250):
        result = paired_recall_test({q: v2_validation[q][:k] for q in ids}, {q: v1_dense[q][:k] for q in ids},
                                    relevant, query_ids=sorted(ids), k=k, n_resamples=20_000, seed=RANDOM_SEED)
        channel_rows.append({"segment": segment, "k": k,
                             "v1": recall_at_k(select(v1_dense, ids), select(relevant, ids), k=k),
                             "v2": recall_at_k(select(v2_validation, ids), select(relevant, ids), k=k),
                             **asdict(result)})
channel_table = pd.DataFrame(channel_rows)
display(channel_table.round(4))

# %% [markdown]
# ## 3. Признаки learned fusion для любого набора каналов
#
# Та же схема признаков, что в notebook 17 (feature set B), но набор
# dense-каналов параметризован. Для каждого dense-канала: ранг в global,
# local и в их RRF; общие признаки — ранги BM25, `min_rank`, `source_count`,
# ранг RRF всех каналов системы, token overlaps, локация, item priors.

# %%
DENSE = {
    "zero": ("dense_global", "dense_local", 1.15),
    "v1": ("finetuned_global", "finetuned_local", 1.5),
    "v2": ("finetuned_v2_global", "finetuned_v2_local", V2_LOCAL_WEIGHT),
}
SYSTEMS = {
    "C1": ("zero", "v1"),
    "C2": ("zero", "v2"),
    "C3": ("zero", "v1", "v2"),
    "C4": ("v2",),
}

channels: dict[tuple[str, str], dict[str, list[str]]] = {}
for row in rankings.itertuples(index=False):
    lists = {"bm25": parse_rank(row.bm25), "bm25_plain": parse_rank(row.bm25_plain)}
    for name, (global_column, local_column, weight) in DENSE.items():
        lists[f"{name}_global"] = parse_rank(getattr(row, global_column))
        lists[f"{name}_local"] = parse_rank(getattr(row, local_column))
        lists[name] = reciprocal_rank_fusion(
            [lists[f"{name}_global"], lists[f"{name}_local"]], weights=[1.0, weight], rrf_k=60, top_k=250)
    channels[(row.split, str(row.query_key))] = lists

queries = pd.concat([
    manifest.assign(query_key=manifest.eval_query_id, split="validation"),
    benchmark_queries.assign(query_key=benchmark_queries.query_id, split="benchmark"),
], ignore_index=True).set_index(["split", "query_key"])

pool_items = {item for lists in channels.values()
              for name in ("bm25", "zero", "v1", "v2") for item in lists[name][:POOL_DEPTH]}
items = pd.read_parquet(ROOT / "dataset/benchmark_items.parquet", columns=[
    "item_id", "item_title_raw", "item_infm_params_text", "item_description_raw",
    "item_location_id", "item_rating", "item_rating_reviews_count", "item_price",
    "item_is_phone_hidden", "item_is_message_forbidden"])
items["item_id"] = items.item_id.astype(str)
items = items[items.item_id.isin(pool_items)].set_index("item_id")
item_cache = {
    item_id: {
        "title": tokens(row.item_title_raw), "params": tokens(row.item_infm_params_text),
        "desc": tokens(row.item_description_raw),
        "title_text": clean(row.item_title_raw), "desc_text": clean(row.item_description_raw),
        "location": int(row.item_location_id),
        "rating": float(row.item_rating) if pd.notna(row.item_rating) else -1.0,
        "log_reviews": math.log1p(safe_nonnegative(row.item_rating_reviews_count)),
        "log_price": math.log1p(safe_nonnegative(row.item_price)),
        "phone_hidden": int(row.item_is_phone_hidden == 1),
        "message_forbidden": int(row.item_is_message_forbidden == 1),
    }
    for item_id, row in items.iterrows()
}


def overlap(query_tokens: frozenset[str], item_tokens: frozenset[str]) -> tuple[int, float, float]:
    shared = len(query_tokens & item_tokens)
    return shared, shared / max(len(query_tokens), 1), shared / max(len(query_tokens | item_tokens), 1)


def system_features(system: str, key: tuple[str, str]) -> pd.DataFrame:
    dense = SYSTEMS[system]
    query, lists = queries.loc[key], channels[key]
    fused = reciprocal_rank_fusion([lists["bm25"], *(lists[name] for name in dense)],
                                   weights=[1.0] * (1 + len(dense)), rrf_k=20, top_k=250)
    rank_lists = {"bm25": lists["bm25"], "bm25_plain": lists["bm25_plain"], "fused": fused}
    for name in dense:
        for suffix in ("", "_global", "_local"):
            rank_lists[f"{name}{suffix}"] = lists[f"{name}{suffix}"]
    ranks = {name: {item: rank for rank, item in enumerate(values, 1)} for name, values in rank_lists.items()}
    sources = ("bm25", *dense)
    pool = list(dict.fromkeys(item for name in sources for item in lists[name][:POOL_DEPTH]))
    q_tokens = tokens(f"{query.search_query} {query.search_infm_params_text}")
    f_tokens = tokens(query.search_infm_params_text)
    q_text = clean(query.search_query)
    rows = []
    for item in pool:
        cached = item_cache[item]
        row = {name: ranks[name].get(item, MISSING_RANK) for name in rank_lists}
        row.update({
            "query_key": key[1], "item_id": item,
            "min_rank": min(row[name] for name in sources),
            "source_count": sum(row[name] <= POOL_DEPTH for name in sources),
            "query_tokens": len(q_tokens),
            "location_match": int(cached["location"] == int(query.search_location_id)),
            "exact_in_title": int(bool(q_text) and q_text in cached["title_text"]),
            "exact_in_desc": int(bool(q_text) and q_text in cached["desc_text"]),
            "title_tokens": len(cached["title"]), "desc_tokens": len(cached["desc"]),
            **{name: cached[name] for name in
               ("rating", "log_reviews", "log_price", "phone_hidden", "message_forbidden")},
        })
        for field in ("title", "params", "desc"):
            row[f"{field}_overlap"], row[f"{field}_coverage"], row[f"{field}_jaccard"] = overlap(
                q_tokens, cached[field])
        row["filter_overlap"], row["filter_coverage"], _ = overlap(f_tokens, cached["params"])
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def feature_columns(system: str) -> list[str]:
    dense = SYSTEMS[system]
    return ["bm25", "bm25_plain", "fused",
            *(f"{name}{suffix}" for name in dense for suffix in ("", "_global", "_local")),
            "min_rank", "source_count", "query_tokens", "location_match", "exact_in_title",
            "exact_in_desc", "title_tokens", "desc_tokens",
            "title_overlap", "title_coverage", "title_jaccard", "params_overlap", "params_coverage",
            "params_jaccard", "desc_overlap", "desc_coverage", "desc_jaccard",
            "filter_overlap", "filter_coverage",
            "rating", "log_reviews", "log_price", "phone_hidden", "message_forbidden"]


def make_pool(frame: pd.DataFrame, features: list[str]) -> Pool:
    group_id = pd.factorize(frame.query_key, sort=False)[0].astype("int32")
    return Pool(frame[features], label=frame.label, group_id=group_id)


def train(frame: pd.DataFrame, features: list[str]) -> CatBoostRanker:
    model = CatBoostRanker(loss_function="YetiRankPairwise", iterations=TREES, depth=6,
                           learning_rate=0.05, random_seed=RANDOM_SEED, thread_count=-1,
                           verbose=False, allow_writing_files=False)
    model.fit(make_pool(frame, features))
    return model


def top50(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, list[str]]:
    ordered = frame.assign(score=scores).sort_values(
        ["query_key", "score", "fused"], ascending=[True, False, True])
    return {query_id: values[:50] for query_id, values in
            ordered.groupby("query_key", sort=False).item_id.agg(list).items()}


features_by_system = {}
for system in SYSTEMS:
    frame = pd.concat([system_features(system, ("validation", q)) for q in manifest.eval_query_id],
                      ignore_index=True)
    frame["label"] = [int(item in relevant[q]) for q, item in zip(frame.query_key, frame.item_id)]
    features_by_system[system] = frame
    print(system, frame.shape, "pool positives:", int(frame.label.sum()))

# %% [markdown]
# ## 4. Cross-fitted выбор на dev и одна проверка на test

# %%
oof = {system: {} for system in SYSTEMS}
for fit_ids, held_ids in cross_folds:
    for system, frame in features_by_system.items():
        columns = feature_columns(system)
        model = train(frame[frame.query_key.isin(fit_ids)], columns)
        held = frame[frame.query_key.isin(held_ids)]
        oof[system].update(top50(held, model.predict(held[columns])))

dev_tail = dev_ids & tail_ids
selection = pd.DataFrame([{
    "system": system, "encoders": len(SYSTEMS[system]),
    "oof_dev_tail": recall_at_k(select(predictions, dev_tail), select(relevant, dev_tail)),
    "oof_dev": recall_at_k(select(predictions, dev_ids), select(relevant, dev_ids)),
} for system, predictions in oof.items()])
display(selection.round(5))
challengers = selection[selection.system.ne("C1")].sort_values(
    ["oof_dev_tail", "oof_dev", "encoders"], ascending=[False, False, True], kind="stable")
selected_system = str(challengers.iloc[0].system)
print("selected on OOF dev-tail:", selected_system)

test_predictions = {}
final_models = {}
for system, frame in features_by_system.items():
    columns = feature_columns(system)
    final_models[system] = train(frame[frame.query_key.isin(dev_ids)], columns)
    held = frame[frame.query_key.isin(test_ids)]
    test_predictions[system] = top50(held, final_models[system].predict(held[columns]))

alpha = 0.05 / 3
report_rows, decision = [], {}
for system in ("C2", "C3", "C4"):
    for segment, ids in {"test_tail": test_ids & tail_ids, "test": test_ids}.items():
        result = paired_recall_test(test_predictions[system], test_predictions["C1"], relevant,
                                    query_ids=sorted(ids), confidence=1 - alpha,
                                    n_resamples=20_000, seed=RANDOM_SEED)
        new = per_query_recall(test_predictions[system], relevant, query_ids=sorted(ids))
        old = per_query_recall(test_predictions["C1"], relevant, query_ids=sorted(ids))
        delta = np.asarray([new[q] - old[q] for q in sorted(ids)])
        report_rows.append({
            "system": system, "segment": segment, "selected": system == selected_system,
            "recall": recall_at_k(select(test_predictions[system], ids), select(relevant, ids)),
            "c1_recall": recall_at_k(select(test_predictions["C1"], ids), select(relevant, ids)),
            **asdict(result), "wins": int((delta > 0).sum()), "losses": int((delta < 0).sum()),
        })
        if system == selected_system:
            decision[segment] = bool(result.p_value_greater < alpha and result.mean_delta > 0)
test_report = pd.DataFrame(report_rows)
display(test_report.round(5))
print({"alpha": alpha, "selected": selected_system, **decision})

# %% [markdown]
# ## 5. Benchmark-кандидат
#
# `answer_lora_v2.csv` пишется всегда; `answer.csv` заменяется, только если
# подтверждён primary или secondary endpoint выбранной системы.

# %%
system = selected_system
columns = feature_columns(system)
benchmark_frame = pd.concat([system_features(system, ("benchmark", str(q))) for q in benchmark_queries.query_id],
                            ignore_index=True)
benchmark_predictions = top50(benchmark_frame, final_models[system].predict(benchmark_frame[columns]))
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
answer.to_csv(ROOT / "answer_lora_v2.csv", index=False)
if decision.get("test_tail") or decision.get("test"):
    answer.to_csv(ROOT / "answer.csv", index=False)
current = pd.read_csv(ROOT / "answer_light_selector.csv", dtype=str).set_index("query_id").answer
overlap_with_candidate_3 = float(np.mean([
    len(set(benchmark_predictions[q]) & set(current[q].split())) / 50 for q in current.index]))
print(f"benchmark overlap with candidate 3: {overlap_with_candidate_3:.3f}")

(ROOT / "reports/lora_v2_metrics.json").write_text(json.dumps({
    "v2_local_weight": V2_LOCAL_WEIGHT, "v2_local_weight_dev": weight_scores,
    "channel": channel_table.to_dict("records"),
    "systems": {name: list(value) for name, value in SYSTEMS.items()},
    "selection": selection.to_dict("records"), "selected": selected_system,
    "bonferroni_alpha": alpha, "test": test_report.to_dict("records"), "decision": decision,
    "benchmark_overlap_with_candidate_3": overlap_with_candidate_3,
}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print("saved reports/lora_v2_metrics.json")
