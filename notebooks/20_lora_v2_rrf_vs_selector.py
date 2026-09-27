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
# # LoRA v2: финальное сравнение RRF и candidate selector
#
# После получения LoRA v2 оставляем только два финальных варианта:
#
# 1. **RRF-only** — детерминированное объединение BM25, zero-shot dense,
#    LoRA v1 и LoRA v2 без обучаемой модели;
# 2. **candidate selector** — выбранная в notebook 19 система C3 над тем же
#    набором каналов.
#
# Цель notebook — получить интерпретируемый RRF-контроль, сохранить его CSV и
# свести результаты с selector в одну таблицу. Веса RRF выбираются только на
# dev-tail (`query_frequency <= 1`), test не используется до окончательного
# выбора. Набор конфигураций небольшой и задан до просмотра результатов ниже.

# %%
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import re
import sys

import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from avito_retrieval.fusion import reciprocal_rank_fusion
from avito_retrieval.metrics import recall_at_k
from avito_retrieval.statistics import paired_recall_test

SEED = 42
RRF_K = 20
TAIL_MAX_FREQUENCY = 1


def parse_rank(value: object) -> list[str]:
    if value is None or pd.isna(value) or not str(value):
        return []
    return str(value).split()


def select(mapping: dict, ids: set[str]) -> dict:
    return {query_id: mapping[query_id] for query_id in ids}


# %% [markdown]
# ## 1. Неизменный split и четыре retrieval-канала
#
# Global/local dense rankings сначала сворачиваются с весами, выбранными в
# прежних notebooks только на dev: zero `1:1.15`, v1 `1:1.5`, v2 `1:2.0`.
# Category, rating и location-local ограничения уже применены внутри исходных
# rankings, поэтому RRF не обходит фильтры.

# %%
manifest = pd.read_parquet(ROOT / "artifacts/validation/manifest.parquet").copy()
labels = pd.read_parquet(ROOT / "artifacts/validation/labels.parquet")
queries = pd.read_parquet(ROOT / "dataset/benchmark_queries.parquet")
items = pd.read_parquet(ROOT / "dataset/benchmark_items.parquet", columns=["item_id"])
relevant = labels.groupby("eval_query_id").item_id.agg(set).to_dict()

manifest["fold"] = -1
outer = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
for fold, (_, indices) in enumerate(outer.split(manifest, manifest.primary_stratum)):
    manifest.loc[indices, "fold"] = fold
dev_ids = set(manifest.loc[manifest.fold.eq(0), "eval_query_id"])
test_ids = set(manifest.loc[manifest.fold.eq(1), "eval_query_id"])
tail_ids = set(manifest.loc[manifest.query_frequency.le(TAIL_MAX_FREQUENCY), "eval_query_id"])

rankings = (
    pd.read_parquet(ROOT / "artifacts/rankings/bm25_rankings.parquet")
    .merge(pd.read_parquet(ROOT / "artifacts/dense_kaggle/dense_rankings.parquet"),
           on=["query_key", "split"], validate="one_to_one")
    .merge(pd.read_parquet(ROOT / "artifacts/finetuned_dense_kaggle/finetuned_dense_rankings.parquet"),
           on=["query_key", "split"], validate="one_to_one")
    .merge(pd.read_parquet(ROOT / "artifacts/finetuned_v2_kaggle/finetuned_v2_dense_rankings.parquet"),
           on=["query_key", "split"], validate="one_to_one")
)


def build_channels(frame: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    result = {}
    for row in frame.itertuples(index=False):
        zero = reciprocal_rank_fusion(
            [parse_rank(row.dense_global), parse_rank(row.dense_local)],
            weights=[1.0, 1.15], rrf_k=60, top_k=250)
        v1 = reciprocal_rank_fusion(
            [parse_rank(row.finetuned_global), parse_rank(row.finetuned_local)],
            weights=[1.0, 1.5], rrf_k=60, top_k=250)
        v2 = reciprocal_rank_fusion(
            [parse_rank(row.finetuned_v2_global), parse_rank(row.finetuned_v2_local)],
            weights=[1.0, 2.0], rrf_k=60, top_k=250)
        result[str(row.query_key)] = {
            "bm25": parse_rank(row.bm25), "zero": zero, "v1": v1, "v2": v2,
        }
    return result


validation_channels = build_channels(rankings[rankings.split.eq("validation")])

# %% [markdown]
# ## 2. Заранее ограниченная family весов
#
# `attempt_2` — прежний RRF-контроль. Остальные варианты проверяют замену v1
# на v2, разделение прежнего dense-веса между v1/v2 и равное объединение всех
# моделей. Это не полный grid search: шесть осмысленных конфигураций снижают
# риск подгонки к 176 dev-tail запросам.

# %%
WEIGHTS: dict[str, tuple[float, float, float, float]] = {
    "attempt_2": (1.0, 0.75, 1.25, 0.0),
    "v2_replace": (1.0, 0.75, 0.0, 1.25),
    "split_v1_v2": (1.0, 0.75, 0.625, 0.625),
    "equal_dense": (1.0, 0.75, 1.0, 1.0),
    "equal_all": (1.0, 1.0, 1.0, 1.0),
    "lexical_heavy": (1.5, 0.75, 0.75, 0.75),
}


def fuse(channels: dict[str, dict[str, list[str]]], weights: tuple[float, ...]) -> dict[str, list[str]]:
    predictions = {}
    for query_id, lists in channels.items():
        active = [(lists[name], weight) for name, weight in
                  zip(("bm25", "zero", "v1", "v2"), weights, strict=True) if weight > 0]
        predictions[query_id] = reciprocal_rank_fusion(
            [ranking for ranking, _ in active], weights=[weight for _, weight in active],
            rrf_k=RRF_K, top_k=50)
    return predictions


prediction_cache = {name: fuse(validation_channels, weights) for name, weights in WEIGHTS.items()}
dev_tail = dev_ids & tail_ids
selection = pd.DataFrame([{
    "method": name,
    "weights": weights,
    "dev_tail": recall_at_k(select(prediction_cache[name], dev_tail), select(relevant, dev_tail)),
    "dev": recall_at_k(select(prediction_cache[name], dev_ids), select(relevant, dev_ids)),
} for name, weights in WEIGHTS.items()])
challengers = selection[selection.method.ne("attempt_2")].sort_values(
    ["dev_tail", "dev"], ascending=[False, False], kind="stable")
selected_name = str(challengers.iloc[0].method)
display(selection.sort_values(["dev_tail", "dev"], ascending=False).round(5))
print("selected RRF:", selected_name, WEIGHTS[selected_name])

# %% [markdown]
# ## 3. Единственная test-оценка RRF
#
# Сравниваем выбранный RRF с отправленным RRF attempt 2. Bonferroni-порог
# `0.05 / 5` учитывает пять challenger-конфигураций. Primary segment —
# benchmark-like test-tail; полный test — secondary.

# %%
alpha = 0.05 / (len(WEIGHTS) - 1)
test_rows = []
for segment, ids in {"test_tail": test_ids & tail_ids, "test": test_ids}.items():
    result = paired_recall_test(
        prediction_cache[selected_name], prediction_cache["attempt_2"], relevant,
        query_ids=sorted(ids), confidence=1 - alpha, n_resamples=20_000, seed=SEED)
    test_rows.append({
        "segment": segment,
        "rrf_v2": recall_at_k(select(prediction_cache[selected_name], ids), select(relevant, ids)),
        "attempt_2": recall_at_k(select(prediction_cache["attempt_2"], ids), select(relevant, ids)),
        **asdict(result),
    })
test_table = pd.DataFrame(test_rows)
display(test_table.round(5))

# %% [markdown]
# ## 4. Два финальных CSV и компактное сравнение
#
# Selector CSV уже создан notebook 19. Здесь материализуется только RRF CSV;
# `answer.csv` остаётся selector-кандидатом, если его offline Recall выше.
# Оба файла независимо проверяются по контракту задачи.

# %%
benchmark_channels = build_channels(rankings[rankings.split.eq("benchmark")])
benchmark_predictions = fuse(benchmark_channels, WEIGHTS[selected_name])
rrf_answer = pd.DataFrame({
    "query_id": queries.query_id.astype(str),
    "answer": [" ".join(benchmark_predictions[str(query_id)]) for query_id in queries.query_id],
})
corpus_ids = set(items.item_id.astype(str))


def validate_answer(frame: pd.DataFrame) -> None:
    assert list(frame.columns) == ["query_id", "answer"]
    assert len(frame) == len(queries) == frame.query_id.nunique()
    assert set(frame.query_id) == set(queries.query_id.astype(str))
    for value in frame.answer:
        item_ids = value.split()
        assert len(item_ids) == 50 == len(set(item_ids))
        assert all(re.fullmatch(r"[0-9a-f]{16}", item_id) for item_id in item_ids)
        assert set(item_ids) <= corpus_ids


validate_answer(rrf_answer)
rrf_path = ROOT / "answer_lora_v2_rrf.csv"
rrf_answer.to_csv(rrf_path, index=False)
selector_path = ROOT / "answer_lora_v2.csv"
selector_answer = pd.read_csv(selector_path, dtype=str)
validate_answer(selector_answer)

selector_report = json.loads((ROOT / "reports/lora_v2_metrics.json").read_text())
selector_sanity = {row["segment"]: row for row in selector_report["actual_attempt_3_sanity_test"]}
rrf_test = test_table.set_index("segment")
final_comparison = [
    {
        "method": "RRF-only", "file": rrf_path.name,
        "test_tail": float(rrf_test.loc["test_tail", "rrf_v2"]),
        "test": float(rrf_test.loc["test", "rrf_v2"]),
        "baseline": "attempt_2 RRF",
        "tail_delta": float(rrf_test.loc["test_tail", "mean_delta"]),
        "test_delta": float(rrf_test.loc["test", "mean_delta"]),
        "sha256": hashlib.sha256(rrf_path.read_bytes()).hexdigest(),
    },
    {
        "method": "candidate selector", "file": selector_path.name,
        "test_tail": float(selector_sanity["test_tail"]["candidate"]),
        "test": float(selector_sanity["test"]["candidate"]),
        "baseline": "actual public attempt 3 selector",
        "tail_delta": float(selector_sanity["test_tail"]["mean_delta"]),
        "test_delta": float(selector_sanity["test"]["mean_delta"]),
        "sha256": hashlib.sha256(selector_path.read_bytes()).hexdigest(),
    },
]
comparison = pd.DataFrame(final_comparison).sort_values(["test_tail", "test"], ascending=False)
display(comparison)

report = {
    "rrf_k": RRF_K,
    "rrf_configs": {name: list(weights) for name, weights in WEIGHTS.items()},
    "selected_rrf_on_dev_tail": selected_name,
    "rrf_selection": selection.to_dict("records"),
    "rrf_test": test_table.to_dict("records"),
    "final_comparison": final_comparison,
    "recommended_for_public_attempt": str(comparison.iloc[0].method),
}
(ROOT / "reports/lora_v2_final_comparison.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print(json.dumps({
    "recommended": report["recommended_for_public_attempt"],
    "rrf_file": str(rrf_path), "selector_file": str(selector_path),
}, ensure_ascii=False, indent=2))
