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
# # PU-selector без LoRA v1: top-200 → top-50
#
# Notebook 23 подтвердил направление на изолированном BM25 + LoRA v2 пуле:
# ограничение числа unlabeled negatives улучшило Recall@50 на test и tail, но
# не обошло лучшую публичную attempt 4. Здесь тот же заранее заданный PU-
# протокол переносится на трёхканальный набор без LoRA v1:
#
# `BM25 + zero-shot BGE-M3 + LoRA v2`.
#
# Для буквального сжатия 200→50 исходный union сначала ограничивается top-200
# по равновесному RRF-признаку `fused`. Все признаки, split и baseline берутся
# из notebook 19. Notebook 19 запускается как воспроизводимая prerequisite-
# стадия: это дороже повторного чтения локального cache, но исключает скрытый
# бинарный артефакт и гарантирует идентичность attempt 4.
#
# ## Gate
#
# Конфигурация выбирается только по OOF dev-tail/dev. На test выбранный
# challenger сравнивается paired bootstrap непосредственно с C3 attempt 4,
# хотя сам обучается на C2 без LoRA v1.
# `answer.csv` можно заменить только при неотрицательном tail и статистически
# положительном primary или secondary endpoint (`alpha=.025`).

# %%
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent

# Выполняем зафиксированный pipeline attempt 4 и получаем feature frames,
# OOF/test predictions и benchmark frame в текущем namespace.
get_ipython().run_line_magic("run", str(ROOT / "notebooks/19_lora_v2_evaluation.py"))

PU_BUDGETS_BEST = (24, 48, 80)
PU_BAG_SEEDS = (41, 42, 43)
PU_RRF_RESERVES = (0, 5, 10)
TOP_POOL = 200
PU_ALPHA = 0.025

# %% [markdown]
# ## 1. Top-200 и доступный oracle ceiling
#
# `fused` — RRF-ранг трёх retrieval-каналов. Ограничение применяется
# одинаково к validation и benchmark и не использует labels.

# %%
C2_FEATURES = feature_columns("C2")
c2_all = features_by_system["C2"].copy()
c2 = c2_all[c2_all.fused.le(TOP_POOL)].copy()
c2 = c2.sort_values(["query_key", "fused"], kind="stable").reset_index(drop=True)

rrf_validation = (
    c2.sort_values(["query_key", "fused"], kind="stable")
    .groupby("query_key", sort=False).item_id.agg(list).to_dict()
)
pool_rows = []
for segment, ids in {"dev": dev_ids, "test_tail": test_ids & tail_ids, "test": test_ids}.items():
    sizes = c2[c2.query_key.isin(ids)].groupby("query_key").size()
    pool_rows.append({
        "segment": segment,
        "mean_pool_size": float(sizes.mean()),
        "p95_pool_size": float(sizes.quantile(.95)),
        "pool_recall_at_200": recall_at_k(select(rrf_validation, ids), select(relevant, ids), k=200),
    })
pool_table = pd.DataFrame(pool_rows)
display(pool_table.round(6))

# %% [markdown]
# ## 2. PU sampling и bagging
#
# Код намеренно совпадает с notebook 23. В каждом bag сохраняются все known
# positives, 2/3 бюджета берутся среди hard candidates (`fused<=75` или один
# из channel ranks <=40), остальное — случайные unlabeled. Score-масштабы
# моделей нормализуются reciprocal rank внутри каждого запроса.

# %%
def pu_make_pool(frame: pd.DataFrame) -> Pool:
    group_id = pd.factorize(frame.query_key, sort=False)[0].astype("int32")
    return Pool(frame[C2_FEATURES], label=frame.label, group_id=group_id)


def pu_train(frame: pd.DataFrame, seed: int) -> CatBoostRanker:
    model = CatBoostRanker(
        loss_function="YetiRankPairwise", iterations=TREES, depth=6,
        learning_rate=0.05, random_seed=seed, thread_count=-1,
        verbose=False, allow_writing_files=False,
    )
    model.fit(pu_make_pool(frame))
    return model


def pu_sample_best(frame: pd.DataFrame, negative_budget: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pieces = []
    hard_budget = int(round(negative_budget * 2 / 3))
    dense_rank_columns = ["zero", "v2"]
    for _, group in frame.groupby("query_key", sort=False):
        positive = group[group.label.eq(1)]
        negative = group[group.label.eq(0)]
        min_channel_rank = negative[["bm25", *dense_rank_columns]].min(axis=1)
        hard = negative[negative.fused.le(75) | min_channel_rank.le(40)]
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


def pu_rank_scores(frame: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    work = pd.DataFrame({"query_key": frame.query_key.to_numpy(), "score": raw})
    ranks = work.groupby("query_key", sort=False).score.rank(method="first", ascending=False)
    return (1.0 / (20.0 + ranks.to_numpy())).astype("float64")


def pu_ensemble_scores(models: list[CatBoostRanker], frame: pd.DataFrame) -> np.ndarray:
    return np.mean([
        pu_rank_scores(frame, model.predict(frame[C2_FEATURES])) for model in models
    ], axis=0)


def pu_order(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, list[str]]:
    return (
        frame.assign(score=scores)
        .sort_values(["query_key", "score", "fused"],
                     ascending=[True, False, True], kind="stable")
        .groupby("query_key", sort=False).item_id.agg(list).to_dict()
    )


def pu_reserve(
    selector: dict[str, list[str]], rrf_order: dict[str, list[str]],
    query_ids: set[str], reserve: int,
) -> dict[str, list[str]]:
    if reserve == 0:
        return {query_id: selector[query_id][:50] for query_id in query_ids}
    output = {}
    for query_id in query_ids:
        chosen = list(selector[query_id][:50 - reserve])
        seen = set(chosen)
        for item in rrf_order[query_id]:
            if item not in seen:
                chosen.append(item)
                seen.add(item)
                if len(chosen) == 50:
                    break
        output[query_id] = chosen
    return output

# %% [markdown]
# ## 3. OOF-выбор на dev

# %%
pu_oof_best = {
    (budget, reserve): {}
    for budget in PU_BUDGETS_BEST for reserve in PU_RRF_RESERVES
}

for fit_ids, held_ids in cross_folds:
    fit = c2[c2.query_key.isin(fit_ids)]
    held = c2[c2.query_key.isin(held_ids)]
    for budget in PU_BUDGETS_BEST:
        models = [
            pu_train(pu_sample_best(fit, budget, seed), seed)
            for seed in PU_BAG_SEEDS
        ]
        ranking = pu_order(held, pu_ensemble_scores(models, held))
        for reserve in PU_RRF_RESERVES:
            pu_oof_best[(budget, reserve)].update(
                pu_reserve(ranking, rrf_validation, held_ids, reserve)
            )

selection_best = pd.DataFrame([{
    "negative_budget": budget,
    "rrf_reserve": reserve,
    "oof_dev_tail": recall_at_k(select(predictions, dev_ids & tail_ids),
                                 select(relevant, dev_ids & tail_ids)),
    "oof_dev": recall_at_k(select(predictions, dev_ids), select(relevant, dev_ids)),
} for (budget, reserve), predictions in pu_oof_best.items()]).sort_values(
    ["oof_dev_tail", "oof_dev", "negative_budget", "rrf_reserve"],
    ascending=[False, False, True, True], kind="stable",
)
display(selection_best.round(6))
best_budget = int(selection_best.iloc[0].negative_budget)
best_reserve = int(selection_best.iloc[0].rrf_reserve)
print("selected", best_budget, best_reserve)

# %% [markdown]
# ## 4. Единственная test-проверка против attempt 4

# %%
dev_c2 = c2[c2.query_key.isin(dev_ids)]
test_c2 = c2[c2.query_key.isin(test_ids)]
best_models = [
    pu_train(pu_sample_best(dev_c2, best_budget, seed), seed)
    for seed in PU_BAG_SEEDS
]
best_ranking = pu_order(test_c2, pu_ensemble_scores(best_models, test_c2))
best_test = pu_reserve(best_ranking, rrf_validation, test_ids, best_reserve)
attempt4_test = test_predictions["C3"]

test_best_rows = []
for segment, ids in {"test_tail": test_ids & tail_ids, "test": test_ids}.items():
    result = paired_recall_test(
        best_test, attempt4_test, relevant, query_ids=sorted(ids), k=50,
        confidence=0.975, n_resamples=20_000, seed=RANDOM_SEED,
    )
    new = per_query_recall(best_test, relevant, query_ids=sorted(ids))
    old = per_query_recall(attempt4_test, relevant, query_ids=sorted(ids))
    differences = np.asarray([new[q] - old[q] for q in sorted(ids)])
    test_best_rows.append({
        "segment": segment,
        "pu_recall": recall_at_k(select(best_test, ids), select(relevant, ids)),
        "attempt4_recall": recall_at_k(select(attempt4_test, ids), select(relevant, ids)),
        **asdict(result),
        "wins": int((differences > 0).sum()),
        "losses": int((differences < 0).sum()),
    })
test_best = pd.DataFrame(test_best_rows)
display(test_best.round(6))

test_lookup = test_best.set_index("segment")
tail_nonnegative = bool(test_lookup.loc["test_tail", "mean_delta"] >= 0)
primary = bool(test_lookup.loc["test_tail", "mean_delta"] > 0
               and test_lookup.loc["test_tail", "p_value_greater"] < PU_ALPHA)
secondary = bool(test_lookup.loc["test", "mean_delta"] > 0
                 and test_lookup.loc["test", "p_value_greater"] < PU_ALPHA)
accepted_best = bool(tail_nonnegative and (primary or secondary))
decision_best = {
    "tail_nonnegative": tail_nonnegative,
    "primary_significant": primary,
    "secondary_significant": secondary,
    "accepted": accepted_best,
}
print(decision_best)

# %% [markdown]
# ## 5. Benchmark candidate
#
# Модель и конфигурация уже зафиксированы до benchmark. CSV сохраняется всегда,
# но корневой лучший ответ меняется только при прохождении test gate.

# %%
benchmark_c2_all = pd.concat([
    system_features("C2", ("benchmark", str(query_id)))
    for query_id in benchmark_queries.query_id
], ignore_index=True)
benchmark_c2 = benchmark_c2_all[benchmark_c2_all.fused.le(TOP_POOL)].copy()
benchmark_c2 = benchmark_c2.sort_values(["query_key", "fused"], kind="stable").reset_index(drop=True)
rrf_benchmark = (
    benchmark_c2.sort_values(["query_key", "fused"], kind="stable")
    .groupby("query_key", sort=False).item_id.agg(list).to_dict()
)
benchmark_best_ranking = pu_order(
    benchmark_c2, pu_ensemble_scores(best_models, benchmark_c2)
)
benchmark_ids = set(benchmark_queries.query_id.astype(str))
benchmark_best = pu_reserve(
    benchmark_best_ranking, rrf_benchmark, benchmark_ids, best_reserve
)
answer_best = pd.DataFrame({
    "query_id": benchmark_queries.query_id.astype(str),
    "answer": [" ".join(benchmark_best[str(q)]) for q in benchmark_queries.query_id],
})
corpus_ids = set(pd.read_parquet(
    ROOT / "dataset/benchmark_items.parquet", columns=["item_id"]
).item_id.astype(str))
assert list(answer_best.columns) == ["query_id", "answer"]
assert len(answer_best) == len(benchmark_queries) == answer_best.query_id.nunique()
for value in answer_best.answer:
    item_ids = value.split()
    assert len(item_ids) == 50 == len(set(item_ids))
    assert all(re.fullmatch(r"[0-9a-f]{16}", item_id) for item_id in item_ids)
    assert set(item_ids) <= corpus_ids

candidate_best_path = ROOT / "answer_pu_best_pool_candidate.csv"
answer_best.to_csv(candidate_best_path, index=False)
candidate_best_hash = hashlib.sha256(candidate_best_path.read_bytes()).hexdigest()
if accepted_best:
    answer_best.to_csv(ROOT / "answer.csv", index=False)

report_best = {
    "protocol": {
        "channels": ["bm25", "zero", "v2"],
        "top_pool": TOP_POOL,
        "pu_budgets": list(PU_BUDGETS_BEST),
        "bag_seeds": list(PU_BAG_SEEDS),
        "rrf_reserves": list(PU_RRF_RESERVES),
    },
    "pool": pool_table.to_dict("records"),
    "selection": selection_best.to_dict("records"),
    "selected": {"negative_budget": best_budget, "rrf_reserve": best_reserve},
    "test_vs_attempt4": test_best.to_dict("records"),
    "decision": decision_best,
    "candidate": {"file": candidate_best_path.name, "sha256": candidate_best_hash},
}
(ROOT / "reports/pu_best_pool_metrics.json").write_text(
    json.dumps(report_best, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
)
print("saved", candidate_best_path.name, candidate_best_hash)
print("answer.csv replaced:", accepted_best)
