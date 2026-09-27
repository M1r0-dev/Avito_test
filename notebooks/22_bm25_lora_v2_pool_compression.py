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
# # BM25 + LoRA v2: Recall@200 и разрыв с Recall@50
#
# Публичная абляция attempt 5 оставила только BM25 и LoRA v2, сохранив
# CatBoost selector и структурные признаки. Она получила Recall@50 `0.829627`
# против `0.837229` у attempt 4. Этот notebook отвечает на следующий вопрос:
# **релевантные документы не найдены или selector не умеет выбрать 50 из
# найденного пула?**
#
# Мы различаем три величины:
#
# 1. `RRF@k` — обычный упорядоченный результат BM25 + LoRA v2;
# 2. `pool Recall@budget` — покрытие объединения top-N каждого канала; для
#    N=100 в пуле не более 200 документов;
# 3. `oracle compression@50` — верхняя граница selector'а: сколько recall
#    можно сохранить, если идеально выбрать до 50 документов из пула.
#
# Порядок внутри pool не используется, поэтому это не «подглядывающий» метод
# формирования submission, а диагностика доступного selector'у множества.
# Все параметры и split полностью повторяют notebook 19.

# %%
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from avito_retrieval.fusion import reciprocal_rank_fusion
from avito_retrieval.metrics import recall_at_k
from avito_retrieval.statistics import paired_recall_test

RANDOM_SEED = 42
TAIL_MAX_FREQUENCY = 1
V2_LOCAL_WEIGHT = 2.0  # selected on dev in notebook 19
SOURCE_DEPTHS = (25, 50, 75, 100, 125)


def parse_rank(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)) or not str(value):
        return []
    return str(value).split()


def subset(mapping: dict[str, object], ids: set[str]) -> dict[str, object]:
    return {query_id: mapping[query_id] for query_id in ids}


def union_pool(first: list[str], second: list[str], depth: int) -> list[str]:
    """Stable union; the order is irrelevant when the full pool is evaluated."""
    return list(dict.fromkeys([*first[:depth], *second[:depth]]))


def oracle_compression_recall(
    pools: dict[str, list[str]],
    relevant: dict[str, set[str]],
    query_ids: set[str],
    output_k: int = 50,
) -> float:
    values = []
    for query_id in query_ids:
        found = len(set(pools[query_id]) & relevant[query_id])
        values.append(min(found, output_k) / len(relevant[query_id]))
    return float(np.mean(values))


# %% [markdown]
# ## 1. Тот же leakage-safe holdout, что в attempt 5
#
# Split не перевыбирается по результатам этого анализа. `test-tail` содержит
# запросы, встречавшиеся в train не более одного раза, и лучше приближает
# benchmark по частотности запросов.

# %%
manifest = pd.read_parquet(ROOT / "artifacts/validation/manifest.parquet").copy()
labels = pd.read_parquet(ROOT / "artifacts/validation/labels.parquet")
relevant = labels.groupby("eval_query_id").item_id.agg(set).to_dict()

manifest["fold"] = -1
outer = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_SEED)
for fold, (_, indices) in enumerate(outer.split(manifest, manifest.primary_stratum)):
    manifest.loc[indices, "fold"] = fold

dev_ids = set(manifest.loc[manifest.fold.eq(0), "eval_query_id"])
test_ids = set(manifest.loc[manifest.fold.eq(1), "eval_query_id"])
tail_ids = set(manifest.loc[manifest.query_frequency.le(TAIL_MAX_FREQUENCY), "eval_query_id"])
segments = {
    "dev": dev_ids,
    "test": test_ids,
    "test_tail": test_ids & tail_ids,
    "test_head": test_ids - tail_ids,
}
pd.Series({name: len(ids) for name, ids in segments.items()}, name="queries")

# %% [markdown]
# ## 2. Восстановление двух retrieval-каналов
#
# BM25 здесь — основной вариант с query filters. LoRA v2 объединяет global и
# location-local поиск через RRF с весом local `2.0`, выбранным только на dev в
# notebook 19. Никаких zero-shot или LoRA v1 rankings в расчёт не входит.

# %%
rankings = pd.read_parquet(ROOT / "artifacts/rankings/bm25_rankings.parquet").merge(
    pd.read_parquet(
        ROOT / "artifacts/finetuned_v2_kaggle/finetuned_v2_dense_rankings.parquet"
    ),
    on=["query_key", "split"],
    validate="one_to_one",
)
validation = rankings[rankings.split.eq("validation")]

bm25: dict[str, list[str]] = {}
v2: dict[str, list[str]] = {}
rrf: dict[str, list[str]] = {}
for row in validation.itertuples(index=False):
    query_id = str(row.query_key)
    bm25[query_id] = parse_rank(row.bm25)
    v2[query_id] = reciprocal_rank_fusion(
        [parse_rank(row.finetuned_v2_global), parse_rank(row.finetuned_v2_local)],
        weights=[1.0, V2_LOCAL_WEIGHT],
        rrf_k=60,
        top_k=250,
    )
    rrf[query_id] = reciprocal_rank_fusion(
        [bm25[query_id], v2[query_id]], weights=[1.0, 1.0], rrf_k=20, top_k=250
    )

# %% [markdown]
# ## 3. Обычный ordered Recall@50/100/200
#
# Это качество простой RRF-сортировки. Оно показывает, сколько можно получить
# только увеличением выдачи, но ещё не измеряет полный union двух каналов.

# %%
ordered_rows = []
for segment, ids in segments.items():
    truth = subset(relevant, ids)
    for method, predictions in {"bm25": bm25, "lora_v2": v2, "rrf": rrf}.items():
        for k in (50, 100, 200):
            ordered_rows.append({
                "segment": segment,
                "method": method,
                "k": k,
                "recall": recall_at_k(subset(predictions, ids), truth, k=k),
            })
ordered = pd.DataFrame(ordered_rows)
display(ordered.pivot_table(index=["segment", "method"], columns="k", values="recall").round(5))

# %% [markdown]
# ## 3.1. Может ли простой weighted RRF лучше сжать пул до 50
#
# Равные веса — лишь baseline. До просмотра test перебираем небольшую сетку
# веса LoRA v2 относительно BM25 и `rrf_k`, выбирая сначала Recall@50 на
# dev-tail, затем на всём dev. Это единственный выбор конфигурации; выбранный
# вариант один раз сравнивается с equal-weight RRF на test-tail и test.
# Grid search на dev не требует коррекции test p-value, а два test endpoint
# проверяются при Bonferroni `alpha=.025`.

# %%
DENSE_WEIGHTS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
RRF_K_VALUES = (10, 20, 40, 60, 100)

rrf_candidates: dict[tuple[float, int], dict[str, list[str]]] = {}
rrf_dev_rows = []
for dense_weight in DENSE_WEIGHTS:
    for rrf_k in RRF_K_VALUES:
        predictions = {
            query_id: reciprocal_rank_fusion(
                [bm25[query_id], v2[query_id]],
                weights=[1.0, dense_weight], rrf_k=rrf_k, top_k=250,
            )
            for query_id in relevant
        }
        rrf_candidates[(dense_weight, rrf_k)] = predictions
        rrf_dev_rows.append({
            "dense_weight": dense_weight,
            "rrf_k": rrf_k,
            "dev_tail": recall_at_k(
                subset(predictions, dev_ids & tail_ids), subset(relevant, dev_ids & tail_ids)
            ),
            "dev": recall_at_k(subset(predictions, dev_ids), subset(relevant, dev_ids)),
        })
rrf_dev = pd.DataFrame(rrf_dev_rows).sort_values(
    ["dev_tail", "dev", "dense_weight", "rrf_k"],
    ascending=[False, False, True, True], kind="stable",
)
display(rrf_dev.head(10).round(6))
selected_key = (float(rrf_dev.iloc[0].dense_weight), int(rrf_dev.iloc[0].rrf_k))
selected_rrf = rrf_candidates[selected_key]
print("selected on dev:", selected_key)

rrf_test_rows = []
for segment, ids in {"test_tail": test_ids & tail_ids, "test": test_ids}.items():
    result = paired_recall_test(
        selected_rrf, rrf, relevant, query_ids=sorted(ids), k=50,
        n_resamples=20_000, seed=RANDOM_SEED, confidence=0.975,
    )
    rrf_test_rows.append({
        "segment": segment,
        "selected_recall": recall_at_k(subset(selected_rrf, ids), subset(relevant, ids)),
        "equal_rrf_recall": recall_at_k(subset(rrf, ids), subset(relevant, ids)),
        **asdict(result),
    })
rrf_test = pd.DataFrame(rrf_test_rows)
display(rrf_test.round(6))

# %% [markdown]
# ## 4. Покрытие union-пула и oracle compression@50
#
# Для каждого source depth объединяем top-N BM25 и top-N LoRA v2. При N=100
# это ровно тот тип пула, который получал selector attempt 5: максимум 200
# элементов до дедупликации. `oracle@50` совпадёт с pool recall, если в одном
# запросе найдено не больше 50 релевантных документов; это проверяется явно.

# %%
pools = {
    depth: {query_id: union_pool(bm25[query_id], v2[query_id], depth)
            for query_id in relevant}
    for depth in SOURCE_DEPTHS
}

pool_rows = []
for segment, ids in segments.items():
    truth = subset(relevant, ids)
    for depth, predictions in pools.items():
        sizes = np.asarray([len(predictions[q]) for q in ids])
        budget = 2 * depth
        pool_rows.append({
            "segment": segment,
            "source_depth": depth,
            "max_budget": budget,
            "mean_pool_size": float(sizes.mean()),
            "p95_pool_size": float(np.quantile(sizes, 0.95)),
            "max_pool_size": int(sizes.max()),
            "pool_recall": recall_at_k(subset(predictions, ids), truth, k=budget),
            "oracle_compression_at_50": oracle_compression_recall(predictions, relevant, ids),
        })
pool_table = pd.DataFrame(pool_rows)
display(pool_table.round(5))

# %% [markdown]
# ## 5. Статистическая проверка ценности глубины 200
#
# Primary comparison задаётся до просмотра результата: union top-100 каждого
# канала (budget ≤200) против union top-50 (budget ≤100). Сравнение paired,
# потому что оба метода проверяются на одинаковых запросах. Отдельно показываем
# test-tail и весь test; два endpoint корректируются Bonferroni (`alpha=.025`).

# %%
depth_tests = []
for segment, ids in {"test_tail": segments["test_tail"], "test": test_ids}.items():
    result = paired_recall_test(
        pools[100], pools[50], relevant,
        query_ids=sorted(ids), k=200, n_resamples=20_000,
        seed=RANDOM_SEED, confidence=0.975,
    )
    depth_tests.append({"segment": segment, **asdict(result)})
depth_tests = pd.DataFrame(depth_tests)
display(depth_tests.round(6))

# %% [markdown]
# ## 6. Compression gap
#
# Attempt 5 selector@50 берётся из уже зафиксированного notebook 19. Это только
# описательная разность средних: paired inference требует сохранить его
# per-query predictions и будет выполнен в следующем PU-selector notebook.
# Здесь принимается лишь решение, существует ли достаточный pool headroom.

# %%
attempt5 = json.loads((ROOT / "reports/lora_v2_metrics.json").read_text(encoding="utf-8"))
c4_test = next(row["recall"] for row in attempt5["test"]
               if row["system"] == "C4" and row["segment"] == "test")
c4_tail = next(row["recall"] for row in attempt5["test"]
               if row["system"] == "C4" and row["segment"] == "test_tail")
pool200_lookup = pool_table[pool_table.source_depth.eq(100)].set_index("segment")
compression_gap = {
    "test": float(pool200_lookup.loc["test", "pool_recall"] - c4_test),
    "test_tail": float(pool200_lookup.loc["test_tail", "pool_recall"] - c4_tail),
}
summary = {
    "public_attempt_5_recall_at_50": 0.829627,
    "offline_selector_at_50": {"test": c4_test, "test_tail": c4_tail},
    "pool_at_most_200": {
        segment: {
            key: float(pool200_lookup.loc[segment, key])
            for key in ("mean_pool_size", "p95_pool_size", "max_pool_size", "pool_recall",
                        "oracle_compression_at_50")
        }
        for segment in ("test", "test_tail")
    },
    "compression_gap": compression_gap,
    "weighted_rrf": {
        "selected_dense_weight": selected_key[0],
        "selected_rrf_k": selected_key[1],
        "dev_grid": rrf_dev.to_dict("records"),
        "test_vs_equal_rrf": rrf_test.to_dict("records"),
    },
    "ordered": ordered.to_dict("records"),
    "pool_depths": pool_table.to_dict("records"),
    "depth_200_vs_100_tests": depth_tests.to_dict("records"),
}
display(summary)

(ROOT / "reports/bm25_lora_v2_pool_compression.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
)
print("saved reports/bm25_lora_v2_pool_compression.json")
