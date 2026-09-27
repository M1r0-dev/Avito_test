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
# # Recall headroom после попытки 2 и проверка service-kind фильтра
#
# Попытка 2 (RRF BM25 + zero-shot dense + fine-tuned dense) получила публичный
# Recall@50 `0.821129`. Перед следующей дорогой GPU-гипотезой нужно понять,
# **где** теряются релевантные объявления:
#
# 1. **retrieval miss** — объявления нет ни в одном канале до глубины 250;
#    лечится новыми каналами или адаптацией encoder;
# 2. **selection miss** — объявление есть в каналах, но RRF не поднимает его
#    в top-50; лечится лучшим отбором (reranker), а не новым retrieval.
#
# Отдельно проверяется дешёвая CPU-гипотеза: фильтр `Вид услуги X` из
# `search_infm_params_text` сейчас лишь приклеен к тексту запроса. Если
# использовать его как структурное ограничение, в top-50 может освободиться
# место для подходящих объявлений.
#
# Протокол тот же, что в notebook 14: неизменный holdout, dev/test split с
# `seed=42`, решения принимаются на dev, подтверждаются один раз на test
# paired bootstrap + one-sided randomization.

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
from avito_retrieval.statistics import paired_recall_test, wilson_interval

RANDOM_SEED = 42
TAIL_MAX_FREQUENCY = 1
# Параметры попытки 2 (notebook 14): внешний RRF и внутренние local-веса.
ATTEMPT_2_WEIGHTS = (1.0, 0.75, 1.25)
RRF_K = 20
DEPTHS = (50, 100, 250)


def parse_rank(value: object) -> list[str]:
    """Parse a space-separated ranking stored in parquet."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or not str(value):
        return []
    return str(value).split()


def select(mapping: dict, ids: set[str]) -> dict:
    return {query_id: mapping[query_id] for query_id in ids}


# %% [markdown]
# ## 1. Holdout, split и каналы попытки 2
#
# Каналы собираются ровно как в notebook 14: dense global/local сворачиваются
# внутри модели (`1:1.15` zero-shot, `1:1.5` fine-tuned, выбраны в notebooks
# 05 и 08), BM25 уже filter-aware. SPLADE добавлен только для диагностики
# oracle-покрытия: в submission он не входит.

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
tail_ids = set(
    manifest.loc[manifest.query_frequency.le(TAIL_MAX_FREQUENCY), "eval_query_id"]
)

rankings = (
    pd.read_parquet(ROOT / "artifacts/rankings/bm25_rankings.parquet")
    .merge(pd.read_parquet(ROOT / "artifacts/dense_kaggle/dense_rankings.parquet"),
           on=["query_key", "split"], validate="one_to_one")
    .merge(pd.read_parquet(
        ROOT / "artifacts/finetuned_dense_kaggle/finetuned_dense_rankings.parquet"),
        on=["query_key", "split"], validate="one_to_one")
    .merge(pd.read_parquet(ROOT / "artifacts/splade_kaggle/splade_rankings.parquet"),
           on=["query_key", "split"], validate="one_to_one")
)
validation = rankings[rankings.split.eq("validation")]

channels: dict[str, dict[str, list[str]]] = {}
for row in validation.itertuples(index=False):
    channels[str(row.query_key)] = {
        "bm25": parse_rank(row.bm25),
        "zero": reciprocal_rank_fusion(
            [parse_rank(row.dense_global), parse_rank(row.dense_local)],
            weights=[1.0, 1.15], rrf_k=60, top_k=250,
        ),
        "fine": reciprocal_rank_fusion(
            [parse_rank(row.finetuned_global), parse_rank(row.finetuned_local)],
            weights=[1.0, 1.5], rrf_k=60, top_k=250,
        ),
        # SPLADE: лучшая конфигурация notebook 06 (q32/d192, local weight 2.0).
        "splade": reciprocal_rank_fusion(
            [parse_rank(row.splade_q32_d192_global), parse_rank(row.splade_q32_d192_local)],
            weights=[1.0, 2.0], rrf_k=60, top_k=250,
        ),
    }


def attempt_2(lists: dict[str, list[str]]) -> list[str]:
    """Final top-50 of the submitted attempt 2."""
    return reciprocal_rank_fusion(
        [lists["bm25"], lists["zero"], lists["fine"]],
        weights=list(ATTEMPT_2_WEIGHTS), rrf_k=RRF_K, top_k=50,
    )


final = {query_id: attempt_2(lists) for query_id, lists in channels.items()}
assert set(final) == set(relevant)

# %% [markdown]
# ## 2. Oracle-покрытие каналов
#
# Oracle@d — доля позитивов, попавших хотя бы в один из выбранных каналов на
# глубине `d`. Это верхняя граница Recall для любого отбора из такого пула:
# если позитива нет в пуле, никакой reranker его не вернёт.

# %%
def oracle_recall(ids: set[str], names: list[str], depth: int) -> float:
    pool = {
        query_id: list(dict.fromkeys(
            item for name in names for item in channels[query_id][name][:depth]
        ))
        for query_id in ids
    }
    return recall_at_k(pool, select(relevant, ids), k=10**9)


segments = {"test": test_ids, "test_tail": test_ids & tail_ids, "all": dev_ids | test_ids}
coverage_rows = []
for segment, ids in segments.items():
    coverage_rows.append({
        "segment": segment, "pool": "attempt_2 top-50", "depth": 50,
        "recall": recall_at_k(select(final, ids), select(relevant, ids)),
    })
    for names in (["bm25"], ["zero"], ["fine"], ["splade"],
                  ["bm25", "zero", "fine"], ["bm25", "zero", "fine", "splade"]):
        for depth in DEPTHS:
            coverage_rows.append({
                "segment": segment, "pool": "+".join(names), "depth": depth,
                "recall": oracle_recall(ids, names, depth),
            })
coverage = pd.DataFrame(coverage_rows)
display(coverage.pivot_table(index=["segment", "pool"], columns="depth", values="recall").round(4))

# %% [markdown]
# ## 3. Где лежат промахи попытки 2
#
# Для каждого позитива вне финального top-50 ищем лучший ранг в каналах
# попытки 2. Доли снабжаются 95% Wilson interval.

# %%
miss_rows = []
for query_id, items in relevant.items():
    for item in items:
        if item in final[query_id]:
            continue
        ranks = {
            name: channels[query_id][name].index(item) + 1
            if item in channels[query_id][name] else np.nan
            for name in ("bm25", "zero", "fine", "splade")
        }
        miss_rows.append({"query_id": query_id, "tail": query_id in tail_ids, **ranks})
misses = pd.DataFrame(miss_rows)
in_pool = misses[["bm25", "zero", "fine"]].notna().any(axis=1)
total_positives = sum(len(items) for items in relevant.values())
miss_summary = {
    "positives": total_positives,
    "missed": len(misses),
    "selection_miss_share": float(in_pool.mean()),
    "selection_miss_wilson_95": wilson_interval(int(in_pool.sum()), len(misses)),
    "retrieval_miss_share": float((~in_pool).mean()),
    "splade_only_share": float((~in_pool & misses.splade.notna()).mean()),
    "best_rank_of_selection_misses": misses.loc[in_pool, ["bm25", "zero", "fine"]]
        .min(axis=1).describe(percentiles=[0.25, 0.5, 0.75]).round(1).to_dict(),
}
display(miss_summary)

# %% [markdown]
# **Вывод раздела.** Большая часть промахов — selection miss: позитив уже в
# пуле, обычно на рангах 40–120. Поэтому следующая дорогая гипотеза — лучший
# отбор из пула (text-only reranker), а не новый retrieval-канал. SPLADE почти
# не добавляет покрытия поверх трёх каналов, что согласуется с его отклонением
# в notebook 06.

# %% [markdown]
# ## 4. Гипотеза: `Вид услуги` из фильтра как структурное ограничение
#
# Значения `Вид услуги` образуют закрытый словарь. Он извлекается из фильтров
# запросов train и benchmark и затем ищется в `item_infm_params_text` —
# там тот же формат `Вид услуги <значение>`. Сначала проверяем безопасность:
# какая доля позитивов holdout согласована с фильтром запроса.

# %%
train_filters = pd.read_parquet(
    ROOT / "dataset/train.parquet", columns=["search_infm_params_text"]
).search_infm_params_text
benchmark_queries = pd.read_parquet(ROOT / "dataset/benchmark_queries.parquet")
all_filters = pd.concat([train_filters, benchmark_queries.search_infm_params_text]).drop_duplicates()

KIND_PREFIX = "Вид услуги "
# Следующий ключ фильтра заканчивает значение `Вид услуги`.
NEXT_KEYS = ("Тип услуги", "Онлайн-запись", "Кто оказывает", "Место оказания", "Услуга ")


def raw_kind(text: str) -> str | None:
    start = text.find(KIND_PREFIX)
    if start < 0:
        return None
    rest = text[start + len(KIND_PREFIX):]
    cut = min([rest.find(key) for key in NEXT_KEYS if key in rest] or [len(rest)])
    return rest[:cut].strip() or None


# Самые длинные значения проверяются первыми, чтобы префикс не перехватил их.
KINDS = sorted({kind for kind in all_filters.map(raw_kind) if kind}, key=len, reverse=True)


def service_kind(text: object) -> str | None:
    """Return the closed-vocabulary `Вид услуги` value from a params string."""
    if not isinstance(text, str):
        return None
    start = text.find(KIND_PREFIX)
    if start < 0:
        return None
    rest = text[start + len(KIND_PREFIX):]
    return next((kind for kind in KINDS if rest.startswith(kind)), None)


items = pd.read_parquet(
    ROOT / "dataset/benchmark_items.parquet", columns=["item_id", "item_infm_params_text"]
)
item_kind = dict(zip(items.item_id.astype(str), items.item_infm_params_text.map(service_kind)))
query_kind = dict(zip(manifest.eval_query_id, manifest.search_infm_params_text.map(service_kind)))
kind_query_ids = {query_id for query_id, kind in query_kind.items() if kind}

consistency = labels[labels.eval_query_id.isin(kind_query_ids)].copy()
consistency["matches"] = [
    item_kind.get(item) == query_kind[query_id]
    for query_id, item in zip(consistency.eval_query_id, consistency.item_id)
]
kind_safety = {
    "vocabulary_size": len(KINDS),
    "items_with_kind": float(pd.Series(list(item_kind.values())).notna().mean()),
    "validation_queries_with_kind": len(kind_query_ids) / len(manifest),
    "benchmark_queries_with_kind": float(
        benchmark_queries.search_infm_params_text.map(service_kind).notna().mean()
    ),
    "positives_under_kind_filter": len(consistency),
    "positive_kind_match_rate": float(consistency.matches.mean()),
    "positive_kind_match_wilson_95": wilson_interval(
        int(consistency.matches.sum()), len(consistency)
    ),
}
display(kind_safety)

# %% [markdown]
# Фильтр почти безопасен, поэтому проверяем три заранее заданных варианта
# поверх попытки 2 (только для запросов с `Вид услуги`):
#
# - `hard` — оставить в каждом канале только объявления нужного вида; если
#   после fusion меньше 50, добрать из исходного RRF;
# - `hard_keep_missing` — то же, но объявления без распознанного вида не
#   удаляются;
# - `stable_partition` — подходящие объявления переносятся вперёд, остальные
#   идут следом в исходном порядке.
#
# Выбор делается на dev; единственная проверка — выбранный вариант против
# попытки 2 на test, one-sided `alpha=0.05`.

# %%
def kind_variant(mode: str) -> dict[str, list[str]]:
    predictions = {}
    for query_id, lists in channels.items():
        kind = query_kind[query_id]
        base_lists = [lists["bm25"], lists["zero"], lists["fine"]]
        if kind is None or mode == "attempt_2":
            predictions[query_id] = final[query_id]
            continue

        def allowed(item: str) -> bool:
            value = item_kind.get(item)
            return value == kind or (mode == "hard_keep_missing" and value is None)

        if mode.startswith("hard"):
            filtered = [[item for item in ranking if allowed(item)] for ranking in base_lists]
        else:  # stable_partition
            filtered = [
                [item for item in ranking if allowed(item)]
                + [item for item in ranking if not allowed(item)]
                for ranking in base_lists
            ]
        fused = reciprocal_rank_fusion(
            filtered, weights=list(ATTEMPT_2_WEIGHTS), rrf_k=RRF_K, top_k=50
        )
        # Backfill guarantees 50 candidates even for very narrow kinds.
        predictions[query_id] = list(dict.fromkeys([*fused, *final[query_id]]))[:50]
    return predictions


kind_predictions = {
    mode: kind_variant(mode)
    for mode in ("attempt_2", "hard", "hard_keep_missing", "stable_partition")
}
kind_rows = []
for mode, predictions in kind_predictions.items():
    kind_rows.append({
        "mode": mode,
        "dev": recall_at_k(select(predictions, dev_ids), select(relevant, dev_ids)),
        "dev_kind": recall_at_k(select(predictions, dev_ids & kind_query_ids),
                                select(relevant, dev_ids & kind_query_ids)),
        "test": recall_at_k(select(predictions, test_ids), select(relevant, test_ids)),
        "test_tail": recall_at_k(select(predictions, test_ids & tail_ids),
                                 select(relevant, test_ids & tail_ids)),
    })
kind_comparison = pd.DataFrame(kind_rows)
display(kind_comparison.round(5))

challengers = kind_comparison[kind_comparison["mode"].ne("attempt_2")]
selected_mode = str(challengers.sort_values("dev", ascending=False, kind="stable").iloc[0]["mode"])
kind_dev_test = paired_recall_test(
    kind_predictions[selected_mode], final, relevant,
    query_ids=sorted(dev_ids), n_resamples=20_000, seed=RANDOM_SEED,
)
kind_test = paired_recall_test(
    kind_predictions[selected_mode], final, relevant,
    query_ids=sorted(test_ids), n_resamples=20_000, seed=RANDOM_SEED,
)
kind_accepted = bool(kind_dev_test.mean_delta > 0 and kind_test.p_value_greater < 0.05)
print("selected on dev:", selected_mode)
display(pd.DataFrame([asdict(kind_dev_test), asdict(kind_test)], index=["dev", "test"]))
print("accepted:", kind_accepted)

# %% [markdown]
# **Вывод раздела.** Даже лучший на dev вариант не даёт подтверждённого
# прироста: текст фильтра уже входит в запрос BM25 и dense, и каналы и так
# возвращают в основном объявления нужного вида. Гипотеза отклонена, попытка
# на платформе не тратится.

# %%
report = {
    "attempt_2_weights": list(ATTEMPT_2_WEIGHTS),
    "coverage": coverage.to_dict("records"),
    "misses": miss_summary,
    "service_kind": {
        "safety": kind_safety,
        "comparison": kind_comparison.to_dict("records"),
        "selected_on_dev": selected_mode,
        "dev_test": asdict(kind_dev_test),
        "test": asdict(kind_test),
        "accepted": kind_accepted,
    },
}
(ROOT / "reports/recall_headroom_metrics.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
)
print("saved reports/recall_headroom_metrics.json")
