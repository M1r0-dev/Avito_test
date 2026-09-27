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
# # Multi-view retrieval: RRF и candidate selector
#
# Stage 21A добавил три LoRA v2 views к текущим BM25/zero/v1/v2 каналам:
#
# - `pqo`: query-only → длинные passages;
# - `sc`: query+filters → короткие title+params;
# - `sqo`: query-only → короткие title+params.
#
# Проверяются ровно два типа финального решения: RRF и candidate selector.
# Все варианты и правила принятия зафиксированы до чтения validation labels:
#
# 1. global/local weight каждого нового view выбирается на dev-tail из
#    `{1.0, 1.5, 2.0}`, tie-break — весь dev, затем меньший вес;
# 2. RRF выбирается из пяти заранее описанных расширений текущего RRF;
# 3. selector выбирается из четырёх расширений текущей системы через 2-fold
#    cross-fitting на dev;
# 4. выбранный вариант один раз сравнивается с соответствующим baseline на
#    test-tail и полном test, Bonferroni учитывает число challengers;
# 5. `answer.csv` меняется только если selector даёт неотрицательную tail-delta
#    и значимый положительный эффект хотя бы на primary или secondary endpoint.
#
# Такой gate нужен из-за benchmark shift: 72% benchmark-запросов имеют train
# frequency ≤1, а tail holdout содержит лишь 189 test-запросов.

# %%
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
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

SEED = 42
TAIL_MAX_FREQUENCY = 1
POOL_DEPTH = 100
MISSING_RANK = 501
TREES = 300
TOKEN_RE = re.compile(r"(?u)[\w]+(?:[-'][\w]+)*")


def clean(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return " ".join(str(value).replace("ё", "е").lower().split())


def tokens(value: object) -> frozenset[str]:
    return frozenset(TOKEN_RE.findall(clean(value)))


def parse_rank(value: object) -> list[str]:
    if value is None or pd.isna(value) or not str(value):
        return []
    return str(value).split()


def select(mapping: dict, ids: set[str]) -> dict:
    return {query_id: mapping[query_id] for query_id in ids}


def safe_nonnegative(value: object) -> float:
    return max(float(value), 0.0) if value is not None and pd.notna(value) else 0.0


# %% [markdown]
# ## 1. Неизменный split и raw rankings

# %%
manifest = pd.read_parquet(ROOT / "artifacts/validation/manifest.parquet").copy()
labels = pd.read_parquet(ROOT / "artifacts/validation/labels.parquet")
benchmark_queries = pd.read_parquet(ROOT / "dataset/benchmark_queries.parquet")
relevant = labels.groupby("eval_query_id").item_id.agg(set).to_dict()

manifest["fold"] = -1
outer = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
for fold, (_, indices) in enumerate(outer.split(manifest, manifest.primary_stratum)):
    manifest.loc[indices, "fold"] = fold
dev_ids = set(manifest.loc[manifest.fold.eq(0), "eval_query_id"])
test_ids = set(manifest.loc[manifest.fold.eq(1), "eval_query_id"])
tail_ids = set(manifest.loc[manifest.query_frequency.le(TAIL_MAX_FREQUENCY), "eval_query_id"])
dev_tail, test_tail = dev_ids & tail_ids, test_ids & tail_ids

dev_frame = manifest[manifest.fold.eq(0)].reset_index(drop=True)
inner = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
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
    .merge(pd.read_parquet(ROOT / "artifacts/finetuned_v2_kaggle/finetuned_v2_dense_rankings.parquet"),
           on=["query_key", "split"], validate="one_to_one")
    .merge(pd.read_parquet(ROOT / "artifacts/multiview_v2_kaggle/multiview_v2_rankings.parquet"),
           on=["query_key", "split"], validate="one_to_one")
)

# %% [markdown]
# ## 2. Выбор global/local веса каждого view только на dev-tail
#
# Base-веса не перенастраиваются: zero `1.15`, v1 `1.5`, v2 `2.0`. Для нового
# view выбирается только его local weight. Самостоятельный Recall view важен как
# диагностика, но решение принимается далее по итоговой системе.

# %%
RAW_VIEWS = {
    "pqo": ("mv_passage_query_only_global", "mv_passage_query_only_local"),
    "sc": ("mv_short_combined_global", "mv_short_combined_local"),
    "sqo": ("mv_short_query_only_global", "mv_short_query_only_local"),
}
LOCAL_WEIGHTS = (1.0, 1.5, 2.0)
validation_rows = rankings[rankings.split.eq("validation")]


def fused_view(frame: pd.DataFrame, columns: tuple[str, str], weight: float) -> dict[str, list[str]]:
    return {
        str(row.query_key): reciprocal_rank_fusion(
            [parse_rank(getattr(row, columns[0])), parse_rank(getattr(row, columns[1]))],
            weights=[1.0, weight], rrf_k=60, top_k=250)
        for row in frame.itertuples(index=False)
    }


view_weight_rows, selected_local_weights = [], {}
for name, columns in RAW_VIEWS.items():
    candidates = []
    for weight in LOCAL_WEIGHTS:
        pred = fused_view(validation_rows, columns, weight)
        row = {
            "view": name, "local_weight": weight,
            "dev_tail": recall_at_k(select(pred, dev_tail), select(relevant, dev_tail)),
            "dev": recall_at_k(select(pred, dev_ids), select(relevant, dev_ids)),
        }
        candidates.append(row); view_weight_rows.append(row)
    best = sorted(candidates, key=lambda row: (-row["dev_tail"], -row["dev"], row["local_weight"]))[0]
    selected_local_weights[name] = float(best["local_weight"])
view_weight_table = pd.DataFrame(view_weight_rows)
display(view_weight_table.round(5))
print("selected local weights:", selected_local_weights)


def build_channels(frame: pd.DataFrame) -> dict[tuple[str, str], dict[str, list[str]]]:
    result = {}
    for row in frame.itertuples(index=False):
        lists = {"bm25": parse_rank(row.bm25), "bm25_plain": parse_rank(row.bm25_plain)}
        specs = {
            "zero": ("dense_global", "dense_local", 1.15),
            "v1": ("finetuned_global", "finetuned_local", 1.5),
            "v2": ("finetuned_v2_global", "finetuned_v2_local", 2.0),
            **{name: (*RAW_VIEWS[name], selected_local_weights[name]) for name in RAW_VIEWS},
        }
        for name, (global_col, local_col, weight) in specs.items():
            lists[f"{name}_global"] = parse_rank(getattr(row, global_col))
            lists[f"{name}_local"] = parse_rank(getattr(row, local_col))
            lists[name] = reciprocal_rank_fusion(
                [lists[f"{name}_global"], lists[f"{name}_local"]],
                weights=[1.0, weight], rrf_k=60, top_k=250)
        result[(str(row.split), str(row.query_key))] = lists
    return result


channels = build_channels(rankings)

# %% [markdown]
# ## 3. Финальный вариант 1 — multi-view RRF
#
# Текущий RRF использует BM25/zero/v1/v2 с весами `1.5/.75/.75/.75`.
# Challengers добавляют ровно один view либо все views умеренным весом; полный
# grid запрещён, чтобы не подогнать 176 dev-tail запросов.

# %%
RRF_CONFIGS = {
    "base": {"bm25": 1.5, "zero": .75, "v1": .75, "v2": .75},
    "plus_pqo": {"bm25": 1.5, "zero": .75, "v1": .75, "v2": .75, "pqo": .75},
    "plus_sc": {"bm25": 1.5, "zero": .75, "v1": .75, "v2": .75, "sc": .75},
    "plus_sqo": {"bm25": 1.5, "zero": .75, "v1": .75, "v2": .75, "sqo": .75},
    "all_views": {"bm25": 1.5, "zero": .75, "v1": .75, "v2": .75,
                  "pqo": .5, "sc": .5, "sqo": .5},
}


def rrf_predictions(split: str, config: dict[str, float]) -> dict[str, list[str]]:
    return {
        query_id: reciprocal_rank_fusion(
            [lists[name] for name in config], weights=list(config.values()), rrf_k=20, top_k=50)
        for (row_split, query_id), lists in channels.items() if row_split == split
    }


rrf_cache = {name: rrf_predictions("validation", config) for name, config in RRF_CONFIGS.items()}
rrf_selection = pd.DataFrame([{
    "method": name,
    "dev_tail": recall_at_k(select(pred, dev_tail), select(relevant, dev_tail)),
    "dev": recall_at_k(select(pred, dev_ids), select(relevant, dev_ids)),
} for name, pred in rrf_cache.items()])
rrf_challengers = rrf_selection[rrf_selection.method.ne("base")].sort_values(
    ["dev_tail", "dev"], ascending=False, kind="stable")
selected_rrf = str(rrf_challengers.iloc[0].method)
display(rrf_selection.round(5)); print("selected RRF:", selected_rrf)

rrf_alpha = 0.05 / (len(RRF_CONFIGS) - 1)
rrf_tests = {}
for segment, ids in {"test_tail": test_tail, "test": test_ids}.items():
    rrf_tests[segment] = paired_recall_test(
        rrf_cache[selected_rrf], rrf_cache["base"], relevant, query_ids=sorted(ids),
        confidence=1 - rrf_alpha, n_resamples=20_000, seed=SEED)
display(pd.DataFrame({name: asdict(value) for name, value in rrf_tests.items()}).T)

# %% [markdown]
# ## 4. Финальный вариант 2 — multi-view candidate selector
#
# S0 точно повторяет систему попытки 4. Четыре challengers добавляют по одному
# view или все сразу. Feature family, 300 trees и pool depth не перенастраются.
# Candidate pool — union top-100 каналов; labels вне известных кликов остаются
# неполными, поэтому test-tail — primary endpoint.

# %%
SYSTEMS = {
    "S0_base": ("zero", "v1", "v2"),
    "S1_pqo": ("zero", "v1", "v2", "pqo"),
    "S2_sc": ("zero", "v1", "v2", "sc"),
    "S3_sqo": ("zero", "v1", "v2", "sqo"),
    "S4_all": ("zero", "v1", "v2", "pqo", "sc", "sqo"),
}
queries = pd.concat([
    manifest.assign(query_key=manifest.eval_query_id, split="validation"),
    benchmark_queries.assign(query_key=benchmark_queries.query_id, split="benchmark"),
], ignore_index=True).set_index(["split", "query_key"])

pool_items = {
    item for lists in channels.values()
    for name in ("bm25", "zero", "v1", "v2", "pqo", "sc", "sqo")
    for item in lists[name][:POOL_DEPTH]
}
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
    } for item_id, row in items.iterrows()
}


def overlap(query_tokens: frozenset[str], item_tokens: frozenset[str]) -> tuple[int, float, float]:
    shared = len(query_tokens & item_tokens)
    return shared, shared / max(len(query_tokens), 1), shared / max(len(query_tokens | item_tokens), 1)


COMMON_FEATURES = [
    "min_rank", "source_count", "query_tokens", "location_match", "exact_in_title",
    "exact_in_desc", "title_tokens", "desc_tokens", "title_overlap", "title_coverage",
    "title_jaccard", "params_overlap", "params_coverage", "params_jaccard",
    "desc_overlap", "desc_coverage", "desc_jaccard", "filter_overlap", "filter_coverage",
    "rating", "log_reviews", "log_price", "phone_hidden", "message_forbidden",
]


def feature_columns(system: str) -> list[str]:
    dense = SYSTEMS[system]
    return ["bm25", "bm25_plain", "fused",
            *(f"{name}{suffix}" for name in dense for suffix in ("", "_global", "_local")),
            *COMMON_FEATURES]


def system_features(system: str, key: tuple[str, str]) -> pd.DataFrame:
    dense = SYSTEMS[system]
    query, lists = queries.loc[key], channels[key]
    fused = reciprocal_rank_fusion(
        [lists["bm25"], *(lists[name] for name in dense)],
        weights=[1.0] * (1 + len(dense)), rrf_k=20, top_k=250)
    rank_lists = {"bm25": lists["bm25"], "bm25_plain": lists["bm25_plain"], "fused": fused}
    for name in dense:
        for suffix in ("", "_global", "_local"):
            rank_lists[f"{name}{suffix}"] = lists[f"{name}{suffix}"]
    ranks = {name: {item: rank for rank, item in enumerate(values, 1)}
             for name, values in rank_lists.items()}
    sources = ("bm25", *dense)
    pool = list(dict.fromkeys(item for name in sources for item in lists[name][:POOL_DEPTH]))
    q_tokens = tokens(f"{query.search_query} {query.search_infm_params_text}")
    f_tokens, q_text = tokens(query.search_infm_params_text), clean(query.search_query)
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


def make_pool(frame: pd.DataFrame, columns: list[str]) -> Pool:
    group_id = pd.factorize(frame.query_key, sort=False)[0].astype("int32")
    return Pool(frame[columns], label=frame.label, group_id=group_id)


def train(frame: pd.DataFrame, columns: list[str]) -> CatBoostRanker:
    model = CatBoostRanker(
        loss_function="YetiRankPairwise", iterations=TREES, depth=6, learning_rate=.05,
        random_seed=SEED, thread_count=-1, verbose=False, allow_writing_files=False)
    model.fit(make_pool(frame, columns)); return model


def top50(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, list[str]]:
    ordered = frame.assign(score=scores).sort_values(
        ["query_key", "score", "fused"], ascending=[True, False, True])
    return {query_id: values[:50] for query_id, values in
            ordered.groupby("query_key", sort=False).item_id.agg(list).items()}


validation_features = {}
oof = {system: {} for system in SYSTEMS}
for system in SYSTEMS:
    frame = pd.concat(
        [system_features(system, ("validation", q)) for q in manifest.eval_query_id],
        ignore_index=True)
    frame["label"] = [int(item in relevant[q]) for q, item in zip(frame.query_key, frame.item_id)]
    validation_features[system] = frame
    print(system, frame.shape, "pool positives", int(frame.label.sum()))

for fit_ids, held_ids in cross_folds:
    for system, frame in validation_features.items():
        columns = feature_columns(system)
        model = train(frame[frame.query_key.isin(fit_ids)], columns)
        held = frame[frame.query_key.isin(held_ids)]
        oof[system].update(top50(held, model.predict(held[columns])))

selector_selection = pd.DataFrame([{
    "system": system,
    "oof_dev_tail": recall_at_k(select(pred, dev_tail), select(relevant, dev_tail)),
    "oof_dev": recall_at_k(select(pred, dev_ids), select(relevant, dev_ids)),
} for system, pred in oof.items()])
selector_challengers = selector_selection[selector_selection.system.ne("S0_base")].sort_values(
    ["oof_dev_tail", "oof_dev"], ascending=False, kind="stable")
selected_selector = str(selector_challengers.iloc[0].system)
display(selector_selection.round(5)); print("selected selector:", selected_selector)

test_predictions, final_models = {}, {}
for system in ("S0_base", selected_selector):
    frame, columns = validation_features[system], feature_columns(system)
    final_models[system] = train(frame[frame.query_key.isin(dev_ids)], columns)
    held = frame[frame.query_key.isin(test_ids)]
    test_predictions[system] = top50(held, final_models[system].predict(held[columns]))

selector_alpha = 0.05 / (len(SYSTEMS) - 1)
selector_tests, directions = {}, {}
for segment, ids in {"test_tail": test_tail, "test": test_ids}.items():
    result = paired_recall_test(
        test_predictions[selected_selector], test_predictions["S0_base"], relevant,
        query_ids=sorted(ids), confidence=1 - selector_alpha,
        n_resamples=20_000, seed=SEED)
    selector_tests[segment] = result
    new = per_query_recall(test_predictions[selected_selector], relevant, query_ids=sorted(ids))
    old = per_query_recall(test_predictions["S0_base"], relevant, query_ids=sorted(ids))
    delta = np.asarray([new[q] - old[q] for q in sorted(ids)])
    directions[segment] = {"wins": int((delta > 0).sum()), "losses": int((delta < 0).sum())}
display(pd.DataFrame({name: asdict(value) for name, value in selector_tests.items()}).T)
print(directions)

tail_nonnegative = selector_tests["test_tail"].mean_delta >= 0
primary = selector_tests["test_tail"].mean_delta > 0 and selector_tests["test_tail"].p_value_greater < selector_alpha
secondary = selector_tests["test"].mean_delta > 0 and selector_tests["test"].p_value_greater < selector_alpha
accepted = bool(tail_nonnegative and (primary or secondary))
print({"alpha": selector_alpha, "primary": primary, "secondary": secondary, "accepted": accepted})

# %% [markdown]
# ## 5. Benchmark CSV, contract и отчёт
#
# RRF и selector сохраняются раздельно. Selector заменяет `answer.csv` только
# после statistical gate; текущая публичная попытка 4 остаётся в `submissions/`.

# %%
corpus_ids = set(pd.read_parquet(
    ROOT / "dataset/benchmark_items.parquet", columns=["item_id"]).item_id.astype(str))


def write_answer(path: Path, predictions: dict[str, list[str]]) -> tuple[pd.DataFrame, str]:
    answer = pd.DataFrame({
        "query_id": benchmark_queries.query_id.astype(str),
        "answer": [" ".join(predictions[str(q)]) for q in benchmark_queries.query_id],
    })
    assert len(answer) == len(benchmark_queries) == answer.query_id.nunique()
    for value in answer.answer:
        ids = value.split()
        assert len(ids) == 50 == len(set(ids)) and set(ids) <= corpus_ids
        assert all(re.fullmatch(r"[0-9a-f]{16}", item_id) for item_id in ids)
    answer.to_csv(path, index=False)
    return answer, hashlib.sha256(path.read_bytes()).hexdigest()


benchmark_rrf = rrf_predictions("benchmark", RRF_CONFIGS[selected_rrf])
rrf_answer, rrf_sha = write_answer(ROOT / "answer_multiview_rrf.csv", benchmark_rrf)

system = selected_selector
benchmark_frame = pd.concat([
    system_features(system, ("benchmark", str(query_id)))
    for query_id in benchmark_queries.query_id], ignore_index=True)
benchmark_selector = top50(
    benchmark_frame, final_models[system].predict(benchmark_frame[feature_columns(system)]))
selector_answer, selector_sha = write_answer(
    ROOT / "answer_multiview_selector.csv", benchmark_selector)
model_path = ROOT / "models/multiview_selector_candidate.cbm"
final_models[system].save_model(str(model_path))
if accepted:
    selector_answer.to_csv(ROOT / "answer.csv", index=False)

report = {
    "local_weight_selection": view_weight_table.to_dict("records"),
    "selected_local_weights": selected_local_weights,
    "rrf_configs": RRF_CONFIGS, "rrf_selection": rrf_selection.to_dict("records"),
    "selected_rrf": selected_rrf, "rrf_alpha": rrf_alpha,
    "rrf_tests": {name: asdict(value) for name, value in rrf_tests.items()},
    "selector_systems": {name: list(value) for name, value in SYSTEMS.items()},
    "selector_selection": selector_selection.to_dict("records"),
    "selected_selector": selected_selector, "selector_alpha": selector_alpha,
    "selector_tests": {name: asdict(value) for name, value in selector_tests.items()},
    "selector_directions": directions,
    "decision": {"tail_nonnegative": bool(tail_nonnegative), "primary": bool(primary),
                 "secondary": bool(secondary), "accepted": accepted},
    "benchmark": {"rrf_sha256": rrf_sha, "selector_sha256": selector_sha},
}
(ROOT / "reports/multiview_retrieval_metrics.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print(json.dumps({"selected_rrf": selected_rrf, "selected_selector": selected_selector,
                  "accepted": accepted, "rrf_sha": rrf_sha,
                  "selector_sha": selector_sha}, indent=2))
