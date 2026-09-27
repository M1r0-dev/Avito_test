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
# # Positive–unlabeled selector: сжатие BM25 + LoRA v2 из 200 в 50
#
# Notebook 22 показал, что пул attempt 5 уже содержит существенно больше
# релевантных объявлений, чем остаётся после selector:
#
# - test: pool Recall@≤200 `0.91687`, selector Recall@50 `0.85651`;
# - test-tail: `0.97354` против `0.88360`.
#
# Значит, bottleneck — compression, а не retrieval. Обычный ranker размечает
# все непрокликанные документы как отрицательные, хотя train содержит только
# выбранные объявления, а не полную экспертную разметку. Это создаёт false
# negatives. Здесь проверяется **positive–unlabeled bagging**: все известные
# positives сохраняются, а из unlabeled пула в каждом bag берётся ограниченная
# стратифицированная выборка hard/other candidates.
#
# ## Зафиксированный до test протокол
#
# - retrieval и признаки в точности как у `C4` notebook 19: BM25 + LoRA v2,
#   source depth 100, максимум 200 кандидатов;
# - baseline: YetiRankPairwise, 300 trees, все unlabeled как negatives;
# - PU budgets: 24, 48, 80 unlabeled на запрос, три независимых bag seed;
# - 2/3 выборки — hard candidates из верхней части RRF, 1/3 — остальные;
# - score ensemble — средний reciprocal rank моделей внутри каждого запроса,
#   чтобы разные шкалы CatBoost не влияли на bagging;
# - страховка RRF: 0, 5 или 10 мест из 50;
# - конфигурация выбирается только по 2-fold OOF dev-tail, затем OOF dev,
#   tie-break — меньший negative budget и reserve;
# - выбранный вариант единожды сравнивается с hard-negative baseline на
#   test-tail (primary) и test (secondary), Bonferroni `alpha=.025`;
# - `answer.csv` меняется только если test-tail не ухудшен, хотя бы один
#   endpoint статистически положителен и результат не уступает offline attempt 4.

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

RANDOM_SEED = 42
TAIL_MAX_FREQUENCY = 1
POOL_DEPTH = 100
MISSING_RANK = 501
TREES = 300
V2_LOCAL_WEIGHT = 2.0
PU_BUDGETS = (24, 48, 80)
BAG_SEEDS = (41, 42, 43)
RRF_RESERVES = (0, 5, 10)
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


def overlap(query_tokens: frozenset[str], item_tokens: frozenset[str]) -> tuple[int, float, float]:
    shared = len(query_tokens & item_tokens)
    return shared, shared / max(len(query_tokens), 1), shared / max(len(query_tokens | item_tokens), 1)


# %% [markdown]
# ## 1. Holdout и retrieval pool
#
# Split и rankings не пересчитываются. Это позволяет сравнить PU-обучение с
# attempt 5 без изменения retriever или состава признаков.

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

rankings = pd.read_parquet(ROOT / "artifacts/rankings/bm25_rankings.parquet").merge(
    pd.read_parquet(
        ROOT / "artifacts/finetuned_v2_kaggle/finetuned_v2_dense_rankings.parquet"
    ),
    on=["query_key", "split"], validate="one_to_one",
)


def channel_lists(row: object) -> dict[str, list[str]]:
    result = {
        "bm25": parse_rank(row.bm25),
        "bm25_plain": parse_rank(row.bm25_plain),
        "v2_global": parse_rank(row.finetuned_v2_global),
        "v2_local": parse_rank(row.finetuned_v2_local),
    }
    result["v2"] = reciprocal_rank_fusion(
        [result["v2_global"], result["v2_local"]],
        weights=[1.0, V2_LOCAL_WEIGHT], rrf_k=60, top_k=250,
    )
    result["fused"] = reciprocal_rank_fusion(
        [result["bm25"], result["v2"]], weights=[1.0, 1.0], rrf_k=20, top_k=250,
    )
    return result


channels = {
    (row.split, str(row.query_key)): channel_lists(row)
    for row in rankings.itertuples(index=False)
}
queries = pd.concat([
    manifest.assign(query_key=manifest.eval_query_id, split="validation"),
    benchmark_queries.assign(query_key=benchmark_queries.query_id, split="benchmark"),
], ignore_index=True).set_index(["split", "query_key"])

# %% [markdown]
# ## 2. Те же признаки, что у C4 attempt 5
#
# Важно не смешивать изменение objective с feature engineering. Поэтому набор
# полностью повторяет notebook 19: ranks, token overlap, filters, location,
# rating/price/review priors и contact flags.

# %%
all_pool_items = {
    item
    for lists in channels.values()
    for source in ("bm25", "v2")
    for item in lists[source][:POOL_DEPTH]
}
items = pd.read_parquet(ROOT / "dataset/benchmark_items.parquet", columns=[
    "item_id", "item_title_raw", "item_infm_params_text", "item_description_raw",
    "item_location_id", "item_rating", "item_rating_reviews_count", "item_price",
    "item_is_phone_hidden", "item_is_message_forbidden",
])
items["item_id"] = items.item_id.astype(str)
items = items[items.item_id.isin(all_pool_items)].set_index("item_id")
item_cache = {
    item_id: {
        "title": tokens(row.item_title_raw),
        "params": tokens(row.item_infm_params_text),
        "desc": tokens(row.item_description_raw),
        "title_text": clean(row.item_title_raw),
        "desc_text": clean(row.item_description_raw),
        "location": int(row.item_location_id),
        "rating": float(row.item_rating) if pd.notna(row.item_rating) else -1.0,
        "log_reviews": math.log1p(safe_nonnegative(row.item_rating_reviews_count)),
        "log_price": math.log1p(safe_nonnegative(row.item_price)),
        "phone_hidden": int(row.item_is_phone_hidden == 1),
        "message_forbidden": int(row.item_is_message_forbidden == 1),
    }
    for item_id, row in items.iterrows()
}


def query_features(key: tuple[str, str]) -> pd.DataFrame:
    query = queries.loc[key]
    lists = channels[key]
    rank_lists = {name: lists[name] for name in
                  ("bm25", "bm25_plain", "fused", "v2", "v2_global", "v2_local")}
    ranks = {
        name: {item: rank for rank, item in enumerate(values, 1)}
        for name, values in rank_lists.items()
    }
    pool = list(dict.fromkeys([
        *lists["bm25"][:POOL_DEPTH], *lists["v2"][:POOL_DEPTH]
    ]))
    q_tokens = tokens(f"{query.search_query} {query.search_infm_params_text}")
    f_tokens = tokens(query.search_infm_params_text)
    q_text = clean(query.search_query)
    rows = []
    for item in pool:
        cached = item_cache[item]
        row = {name: ranks[name].get(item, MISSING_RANK) for name in rank_lists}
        row.update({
            "query_key": key[1], "item_id": item,
            "min_rank": min(row["bm25"], row["v2"]),
            "source_count": int(row["bm25"] <= POOL_DEPTH) + int(row["v2"] <= POOL_DEPTH),
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
                q_tokens, cached[field]
            )
        row["filter_overlap"], row["filter_coverage"], _ = overlap(f_tokens, cached["params"])
        rows.append(row)
    return pd.DataFrame.from_records(rows)


FEATURES = [
    "bm25", "bm25_plain", "fused", "v2", "v2_global", "v2_local",
    "min_rank", "source_count", "query_tokens", "location_match", "exact_in_title",
    "exact_in_desc", "title_tokens", "desc_tokens",
    "title_overlap", "title_coverage", "title_jaccard", "params_overlap", "params_coverage",
    "params_jaccard", "desc_overlap", "desc_coverage", "desc_jaccard",
    "filter_overlap", "filter_coverage", "rating", "log_reviews", "log_price",
    "phone_hidden", "message_forbidden",
]

validation_frame = pd.concat([
    query_features(("validation", str(query_id))) for query_id in manifest.eval_query_id
], ignore_index=True)
validation_frame["label"] = [
    int(item_id in relevant[query_id])
    for query_id, item_id in zip(validation_frame.query_key, validation_frame.item_id)
]
print("validation pool", validation_frame.shape,
      "positives in pool", int(validation_frame.label.sum()))

# %% [markdown]
# ## 3. Hard-negative baseline и PU sampling
#
# Fixed negative budget также выравнивает вклад запросов: при полном пуле
# запрос с 190 unlabeled создаёт почти вдвое больше пар, чем запрос со 100.
# Hard strata сохраняет трудные документы, а случайная треть не даёт модели
# видеть только ближайших семантических соседей.

# %%
def make_pool(frame: pd.DataFrame) -> Pool:
    group_id = pd.factorize(frame.query_key, sort=False)[0].astype("int32")
    return Pool(frame[FEATURES], label=frame.label, group_id=group_id)


def train(frame: pd.DataFrame, seed: int) -> CatBoostRanker:
    model = CatBoostRanker(
        loss_function="YetiRankPairwise", iterations=TREES, depth=6,
        learning_rate=0.05, random_seed=seed, thread_count=-1,
        verbose=False, allow_writing_files=False,
    )
    model.fit(make_pool(frame))
    return model


def pu_sample(frame: pd.DataFrame, negative_budget: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pieces = []
    hard_budget = int(round(negative_budget * 2 / 3))
    for _, group in frame.groupby("query_key", sort=False):
        positive = group[group.label.eq(1)]
        negative = group[group.label.eq(0)]
        hard_mask = negative.fused.le(75) | negative.min_rank.le(40)
        hard = negative[hard_mask]
        other = negative[~hard_mask]

        take_hard = min(hard_budget, len(hard))
        hard_indices = rng.choice(hard.index.to_numpy(), size=take_hard, replace=False)
        remaining = negative_budget - take_hard
        take_other = min(remaining, len(other))
        other_indices = rng.choice(other.index.to_numpy(), size=take_other, replace=False)

        # If one stratum is small, fill the remainder from unused negatives.
        selected = set(hard_indices.tolist()) | set(other_indices.tolist())
        missing = min(negative_budget - len(selected), len(negative) - len(selected))
        if missing > 0:
            available = negative.index[~negative.index.isin(selected)].to_numpy()
            selected.update(rng.choice(available, size=missing, replace=False).tolist())
        pieces.append(pd.concat([positive, negative.loc[sorted(selected)]], axis=0))
    sampled = pd.concat(pieces, ignore_index=True)
    return sampled.sort_values("query_key", kind="stable").reset_index(drop=True)


def rank_normalized_scores(frame: pd.DataFrame, raw_scores: np.ndarray) -> np.ndarray:
    work = pd.DataFrame({"query_key": frame.query_key.to_numpy(), "score": raw_scores})
    ranks = work.groupby("query_key", sort=False).score.rank(method="first", ascending=False)
    return (1.0 / (20.0 + ranks.to_numpy())).astype("float64")


def ensemble_scores(models: list[CatBoostRanker], frame: pd.DataFrame) -> np.ndarray:
    return np.mean([
        rank_normalized_scores(frame, model.predict(frame[FEATURES]))
        for model in models
    ], axis=0)


def ordered_predictions(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, list[str]]:
    ordered = frame.assign(score=scores).sort_values(
        ["query_key", "score", "fused"], ascending=[True, False, True], kind="stable"
    )
    return ordered.groupby("query_key", sort=False).item_id.agg(list).to_dict()


def reserve_rrf(
    selector: dict[str, list[str]], query_ids: set[str], reserve: int
) -> dict[str, list[str]]:
    if reserve == 0:
        return {query_id: selector[query_id][:50] for query_id in query_ids}
    output = {}
    for query_id in query_ids:
        chosen = list(selector[query_id][:50 - reserve])
        chosen_set = set(chosen)
        for item in channels[("validation", query_id)]["fused"]:
            if item not in chosen_set:
                chosen.append(item)
                chosen_set.add(item)
                if len(chosen) == 50:
                    break
        output[query_id] = chosen
    return output


# %% [markdown]
# ## 4. OOF selection только на dev

# %%
baseline_oof: dict[str, list[str]] = {}
pu_oof = {
    (budget, reserve): {}
    for budget in PU_BUDGETS for reserve in RRF_RESERVES
}

for fold_index, (fit_ids, held_ids) in enumerate(cross_folds):
    fit = validation_frame[validation_frame.query_key.isin(fit_ids)]
    held = validation_frame[validation_frame.query_key.isin(held_ids)]

    baseline_model = train(fit, RANDOM_SEED + fold_index)
    baseline_oof.update(ordered_predictions(held, baseline_model.predict(held[FEATURES])))

    for budget in PU_BUDGETS:
        models = [
            train(pu_sample(fit, budget, seed), seed)
            for seed in BAG_SEEDS
        ]
        ranking = ordered_predictions(held, ensemble_scores(models, held))
        for reserve in RRF_RESERVES:
            pu_oof[(budget, reserve)].update(reserve_rrf(ranking, held_ids, reserve))

dev_tail = dev_ids & tail_ids
selection_rows = [{
    "negative_budget": budget,
    "rrf_reserve": reserve,
    "oof_dev_tail": recall_at_k(select(predictions, dev_tail), select(relevant, dev_tail)),
    "oof_dev": recall_at_k(select(predictions, dev_ids), select(relevant, dev_ids)),
} for (budget, reserve), predictions in pu_oof.items()]
selection = pd.DataFrame(selection_rows).sort_values(
    ["oof_dev_tail", "oof_dev", "negative_budget", "rrf_reserve"],
    ascending=[False, False, True, True], kind="stable",
)
baseline_oof_scores = {
    "oof_dev_tail": recall_at_k(select(baseline_oof, dev_tail), select(relevant, dev_tail)),
    "oof_dev": recall_at_k(select(baseline_oof, dev_ids), select(relevant, dev_ids)),
}
display(selection.round(6))
print("hard baseline", baseline_oof_scores)
selected_budget = int(selection.iloc[0].negative_budget)
selected_reserve = int(selection.iloc[0].rrf_reserve)
print("selected", selected_budget, selected_reserve)

# %% [markdown]
# ## 5. Одна проверка выбранной конфигурации на test
#
# Baseline и challenger обучаются на одинаковом dev и оцениваются на одинаковых
# test-запросах. Кроме средних выводятся wins/losses, paired bootstrap 97.5% CI
# и one-sided randomization p-value.

# %%
dev_training = validation_frame[validation_frame.query_key.isin(dev_ids)]
test_frame = validation_frame[validation_frame.query_key.isin(test_ids)]

baseline_model = train(dev_training, RANDOM_SEED)
baseline_test = ordered_predictions(test_frame, baseline_model.predict(test_frame[FEATURES]))

selected_models = [
    train(pu_sample(dev_training, selected_budget, seed), seed)
    for seed in BAG_SEEDS
]
selected_ranking = ordered_predictions(test_frame, ensemble_scores(selected_models, test_frame))
selected_test = reserve_rrf(selected_ranking, test_ids, selected_reserve)

test_rows = []
for segment, ids in {"test_tail": test_ids & tail_ids, "test": test_ids}.items():
    result = paired_recall_test(
        selected_test, baseline_test, relevant, query_ids=sorted(ids),
        k=50, n_resamples=20_000, seed=RANDOM_SEED, confidence=0.975,
    )
    new = per_query_recall(selected_test, relevant, query_ids=sorted(ids))
    old = per_query_recall(baseline_test, relevant, query_ids=sorted(ids))
    delta = np.asarray([new[q] - old[q] for q in sorted(ids)])
    test_rows.append({
        "segment": segment,
        "pu_recall": recall_at_k(select(selected_test, ids), select(relevant, ids)),
        "hard_baseline_recall": recall_at_k(select(baseline_test, ids), select(relevant, ids)),
        **asdict(result),
        "wins": int((delta > 0).sum()), "losses": int((delta < 0).sum()),
    })
test_report = pd.DataFrame(test_rows)
display(test_report.round(6))

test_by_segment = test_report.set_index("segment")
tail_nonnegative = bool(test_by_segment.loc["test_tail", "mean_delta"] >= 0)
primary = bool(test_by_segment.loc["test_tail", "p_value_greater"] < 0.025
               and test_by_segment.loc["test_tail", "mean_delta"] > 0)
secondary = bool(test_by_segment.loc["test", "p_value_greater"] < 0.025
                 and test_by_segment.loc["test", "mean_delta"] > 0)

# Attempt 4 is the actual deployment baseline, even though this experiment is
# an isolated attempt-5 ablation. Requiring both segment means not to be lower
# avoids spending a public attempt on a candidate that only beats weak C4.
attempt4_floor = {"test_tail": 0.91005291005291, "test": 0.8612017400761283}
beats_attempt4_offline = bool(all(
    test_by_segment.loc[segment, "pu_recall"] >= value
    for segment, value in attempt4_floor.items()
))
accepted = bool(tail_nonnegative and (primary or secondary) and beats_attempt4_offline)
decision = {
    "tail_nonnegative_vs_hard_c4": tail_nonnegative,
    "primary_significant": primary,
    "secondary_significant": secondary,
    "beats_attempt4_offline": beats_attempt4_offline,
    "accepted": accepted,
}
print(decision)

# %% [markdown]
# ## 6. Benchmark candidate и защитный gate
#
# Кандидат сохраняется для аудита всегда. Корневой `answer.csv` заменяется
# только при выполнении gate выше; иначе лучшая публичная попытка 4 остаётся
# нетронутой.

# %%
benchmark_frame = pd.concat([
    query_features(("benchmark", str(query_id))) for query_id in benchmark_queries.query_id
], ignore_index=True)
benchmark_ranking = ordered_predictions(
    benchmark_frame, ensemble_scores(selected_models, benchmark_frame)
)


def reserve_benchmark(selector: dict[str, list[str]], reserve: int) -> dict[str, list[str]]:
    if reserve == 0:
        return {q: values[:50] for q, values in selector.items()}
    output = {}
    for query_id, values in selector.items():
        chosen = list(values[:50 - reserve])
        chosen_set = set(chosen)
        for item in channels[("benchmark", query_id)]["fused"]:
            if item not in chosen_set:
                chosen.append(item)
                chosen_set.add(item)
                if len(chosen) == 50:
                    break
        output[query_id] = chosen
    return output


benchmark_predictions = reserve_benchmark(benchmark_ranking, selected_reserve)
answer = pd.DataFrame({
    "query_id": benchmark_queries.query_id.astype(str),
    "answer": [" ".join(benchmark_predictions[str(q)]) for q in benchmark_queries.query_id],
})
corpus_ids = set(pd.read_parquet(
    ROOT / "dataset/benchmark_items.parquet", columns=["item_id"]
).item_id.astype(str))
assert list(answer.columns) == ["query_id", "answer"]
assert len(answer) == len(benchmark_queries) == answer.query_id.nunique()
for value in answer.answer:
    item_ids = value.split()
    assert len(item_ids) == 50 == len(set(item_ids))
    assert all(re.fullmatch(r"[0-9a-f]{16}", item_id) for item_id in item_ids)
    assert set(item_ids) <= corpus_ids

candidate_path = ROOT / "answer_pu_selector_candidate.csv"
answer.to_csv(candidate_path, index=False)
candidate_sha256 = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
if accepted:
    answer.to_csv(ROOT / "answer.csv", index=False)

report = {
    "protocol": {
        "pool_sources": ["bm25", "lora_v2"],
        "pool_depth_per_source": POOL_DEPTH,
        "trees": TREES,
        "pu_budgets": list(PU_BUDGETS),
        "bag_seeds": list(BAG_SEEDS),
        "rrf_reserves": list(RRF_RESERVES),
    },
    "baseline_oof": baseline_oof_scores,
    "selection": selection.to_dict("records"),
    "selected": {"negative_budget": selected_budget, "rrf_reserve": selected_reserve},
    "test": test_report.to_dict("records"),
    "attempt4_floor": attempt4_floor,
    "decision": decision,
    "candidate": {"file": candidate_path.name, "sha256": candidate_sha256},
}
(ROOT / "reports/pu_selector_metrics.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
)
print("saved", candidate_path.name, candidate_sha256)
print("answer.csv replaced:", accepted)

