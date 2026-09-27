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
# # Distribution-shift audit and robust RRF submission
#
# Публичный Recall@50 финального LTR оказался `0.698370`, тогда как на
# train-derived holdout он был `0.85954`. Формат submission отдельно проверен,
# поэтому следующий эксперимент проверяет не ещё более сложный ranker, а
# гипотезу о сдвиге распределения запросов.
#
# Главный риск supervised LTR: holdout содержит запросы из train и почти целиком
# состоит из одной категории. Benchmark заметно чаще содержит редкие или вообще
# не встречавшиеся тексты. Popularity/history и дообученный encoder могут хорошо
# работать на head-query, но плохо переноситься на этот tail. Поэтому здесь:
#
# 1. измеряем shift до выбора модели;
# 2. используем только независимые retrieval-сигналы BM25, zero-shot dense и
#    fine-tuned dense, без click-history и CatBoost;
# 3. выбираем один из заранее заданных RRF-вариантов только на `dev-tail`;
# 4. один раз проверяем выбор на `test-tail` paired-тестом;
# 5. сохраняем отдельный submission, не уничтожая предыдущую попытку.
#
# Это submission-калибровка после наблюдения публичного domain shift. Её нельзя
# трактовать как новый unbiased estimate качества benchmark: публичный score
# всё равно нужен как внешний сигнал.

# %%
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import re
import shutil
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
from avito_retrieval.statistics import paired_recall_test, per_query_recall

RRF_K = 20
TAIL_MAX_FREQUENCY = 1
RANDOM_SEED = 42


def parse_rank(value: object) -> list[str]:
    """Parse a space-separated ranking stored in parquet."""
    if value is None or pd.isna(value) or not str(value):
        return []
    return str(value).split()


def select(mapping: dict[str, list[str]], ids: set[str]) -> dict[str, list[str]]:
    return {query_id: mapping[query_id] for query_id in ids}


def select_truth(mapping: dict[str, set[str]], ids: set[str]) -> dict[str, set[str]]:
    return {query_id: mapping[query_id] for query_id in ids}


def clean_query(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).replace("ё", "е").lower().split())


# %% [markdown]
# ## 1. Загружаем неизменный holdout и retrieval rankings
#
# Двухфолдовый split воспроизводится с тем же seed и той же стратой, что во всех
# предыдущих notebooks. Это сохраняет сопоставимость метрик. Benchmark labels
# нигде не используются: для него доступны только признаки запросов.

# %%
manifest = pd.read_parquet(ROOT / "artifacts/validation/manifest.parquet").copy()
labels = pd.read_parquet(ROOT / "artifacts/validation/labels.parquet")
train = pd.read_parquet(ROOT / "dataset/train.parquet", columns=["search_query"])
benchmark_queries = pd.read_parquet(ROOT / "dataset/benchmark_queries.parquet")
benchmark_items = pd.read_parquet(
    ROOT / "dataset/benchmark_items.parquet", columns=["item_id"]
)

bm25 = pd.read_parquet(ROOT / "artifacts/rankings/bm25_rankings.parquet")
zero = pd.read_parquet(ROOT / "artifacts/dense_kaggle/dense_rankings.parquet")
fine = pd.read_parquet(
    ROOT / "artifacts/finetuned_dense_kaggle/finetuned_dense_rankings.parquet"
)
rankings = bm25.merge(zero, on=["query_key", "split"], validate="one_to_one").merge(
    fine, on=["query_key", "split"], validate="one_to_one"
)

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
dev_tail_ids = dev_ids & tail_ids
test_tail_ids = test_ids & tail_ids
print({
    "dev": len(dev_ids), "test": len(test_ids),
    "dev_tail": len(dev_tail_ids), "test_tail": len(test_tail_ids),
})

# %% [markdown]
# ## 2. Количественно проверяем distribution shift
#
# `query_frequency` для benchmark считаем только по тексту, нормализованному
# тем же способом. Это не leakage: train доступен при обучении, а признак не
# использует benchmark labels. Нас интересует не точное значение popularity,
# а доля запросов, для которых train вообще не даёт надёжного supervised
# сигнала.

# %%
train_frequency = train.search_query.map(clean_query).value_counts()
benchmark_norm = benchmark_queries.search_query.map(clean_query)
benchmark_frequency = benchmark_norm.map(train_frequency).fillna(0).astype(int)

validation_categories = manifest.search_category.value_counts(normalize=True)
benchmark_categories = benchmark_queries.search_category.value_counts(normalize=True)
category_index = validation_categories.index.union(benchmark_categories.index)
category_tv = 0.5 * (
    validation_categories.reindex(category_index, fill_value=0)
    - benchmark_categories.reindex(category_index, fill_value=0)
).abs().sum()

validation_texts = set(manifest.search_query.map(clean_query))
shift = {
    "validation_category_114_share": float(validation_categories.get(114, 0.0)),
    "benchmark_category_114_share": float(benchmark_categories.get(114, 0.0)),
    "validation_category_0_share": float(validation_categories.get(0, 0.0)),
    "benchmark_category_0_share": float(benchmark_categories.get(0, 0.0)),
    "category_total_variation": float(category_tv),
    "validation_tail_share_frequency_le_1": float(
        manifest.query_frequency.le(1).mean()
    ),
    "benchmark_unseen_query_share": float(benchmark_frequency.eq(0).mean()),
    "benchmark_tail_share_frequency_le_1": float(benchmark_frequency.le(1).mean()),
    "benchmark_text_seen_in_validation_share": float(
        benchmark_norm.isin(validation_texts).mean()
    ),
    "validation_query_chars_median": float(manifest.search_query.str.len().median()),
    "benchmark_query_chars_median": float(benchmark_queries.search_query.str.len().median()),
}
display(pd.Series(shift, name="value").to_frame())

# %% [markdown]
# ## 3. Строим три независимых retrieval-канала
#
# Dense global/local сначала сворачиваются внутри каждой модели. Веса `1:1.15`
# для zero-shot и `1:1.5` для fine-tuned были выбраны в notebooks 05 и 08 только
# на dev. На внешнем уровне RRF не использует несопоставимые cosine/BM25 scores,
# а суммирует reciprocal ranks.
#
# Набор внешних весов фиксируем до просмотра `test-tail`. Он мал и покрывает
# осмысленные режимы: прежние двухканальные baselines, равный ансамбль,
# lexical-heavy и несколько вариантов с умеренным преимуществом fine-tuned.
# Нули означают исключение канала; пустые local lists обрабатываются штатно.

# %%
def build_channels(frame: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    for row in frame.itertuples(index=False):
        zero_dense = reciprocal_rank_fusion(
            [parse_rank(row.dense_global), parse_rank(row.dense_local)],
            weights=[1.0, 1.15], rrf_k=60, top_k=250,
        )
        fine_dense = reciprocal_rank_fusion(
            [parse_rank(row.finetuned_global), parse_rank(row.finetuned_local)],
            weights=[1.0, 1.5], rrf_k=60, top_k=250,
        )
        result[str(row.query_key)] = {
            "bm25": parse_rank(row.bm25),
            "zero": zero_dense,
            "fine": fine_dense,
        }
    return result


def fuse_channels(
    channels: dict[str, dict[str, list[str]]],
    weights: tuple[float, float, float],
    *,
    top_k: int = 50,
) -> dict[str, list[str]]:
    predictions: dict[str, list[str]] = {}
    for query_id, values in channels.items():
        active_rankings: list[list[str]] = []
        active_weights: list[float] = []
        for name, weight in zip(("bm25", "zero", "fine"), weights, strict=True):
            if weight > 0:
                active_rankings.append(values[name])
                active_weights.append(weight)
        predictions[query_id] = reciprocal_rank_fusion(
            active_rankings, weights=active_weights, rrf_k=RRF_K, top_k=top_k
        )
    return predictions


validation_channels = build_channels(rankings[rankings.split.eq("validation")])

# `BM25+fine` — наиболее сильный прежний unsupervised baseline. Остальные
# варианты — заранее заданная робастная family, а не полный перебор сетки.
WEIGHT_CONFIGS: dict[str, tuple[float, float, float]] = {
    "bm25": (1.0, 0.0, 0.0),
    "bm25_zero": (1.0, 1.25, 0.0),
    "bm25_fine": (1.0, 0.0, 2.0),
    "equal_three_way": (1.0, 1.0, 1.0),
    "inherited_three_way": (1.0, 1.25, 2.0),
    "moderate_fine": (1.0, 0.75, 1.25),
    "fine_heavy": (1.0, 0.5, 1.5),
    "lexical_heavy": (1.5, 0.75, 1.0),
}

prediction_cache = {
    name: fuse_channels(validation_channels, weights)
    for name, weights in WEIGHT_CONFIGS.items()
}

# %% [markdown]
# ## 4. Выбор на dev-tail и единственная проверка на test-tail
#
# Primary selection metric — Recall@50 на запросах с train frequency `<=1`.
# Именно этот сегмент ближе к benchmark: там таких запросов более 70%. При
# равенстве используем overall dev Recall как вторичный критерий, затем
# стабильный порядок словаря выше. Это не раскрывает test-tail.
#
# После выбора считаем:
#
# - tail и overall Recall на dev/test;
# - paired delta относительно `BM25+fine`;
# - 99.375% CI и randomization p-value (`0.05 / 8`), то есть Bonferroni-gate
#   по числу заранее рассмотренных конфигураций;
# - wins/ties/losses на query-level, чтобы среднее не скрывало асимметрию.

# %%
rows: list[dict[str, object]] = []
for name, predictions in prediction_cache.items():
    rows.append({
        "method": name,
        "weights": WEIGHT_CONFIGS[name],
        "dev_tail": recall_at_k(
            select(predictions, dev_tail_ids), select_truth(relevant, dev_tail_ids)
        ),
        "dev_all": recall_at_k(
            select(predictions, dev_ids), select_truth(relevant, dev_ids)
        ),
    })
dev_comparison = pd.DataFrame(rows).sort_values(
    ["dev_tail", "dev_all"], ascending=[False, False], kind="stable"
)
display(dev_comparison)

selected_name = str(dev_comparison.iloc[0].method)
selected_predictions = prediction_cache[selected_name]
baseline_predictions = prediction_cache["bm25_fine"]
print("Selected on dev-tail:", selected_name, WEIGHT_CONFIGS[selected_name])

comparison = dev_comparison.copy()
comparison["test_tail"] = comparison.method.map(
    lambda name: recall_at_k(
        select(prediction_cache[name], test_tail_ids),
        select_truth(relevant, test_tail_ids),
    )
)
comparison["test_all"] = comparison.method.map(
    lambda name: recall_at_k(
        select(prediction_cache[name], test_ids), select_truth(relevant, test_ids)
    )
)
display(comparison)

confidence = 1 - 0.05 / len(WEIGHT_CONFIGS)
tail_test = paired_recall_test(
    selected_predictions,
    baseline_predictions,
    relevant,
    query_ids=sorted(test_tail_ids),
    confidence=confidence,
    n_resamples=20_000,
    seed=RANDOM_SEED,
)
overall_test = paired_recall_test(
    selected_predictions,
    baseline_predictions,
    relevant,
    query_ids=sorted(test_ids),
    confidence=confidence,
    n_resamples=20_000,
    seed=RANDOM_SEED,
)

selected_recall = per_query_recall(
    selected_predictions, relevant, query_ids=sorted(test_tail_ids)
)
baseline_recall = per_query_recall(
    baseline_predictions, relevant, query_ids=sorted(test_tail_ids)
)
deltas = np.asarray([
    selected_recall[query_id] - baseline_recall[query_id]
    for query_id in sorted(test_tail_ids)
])
direction = {
    "wins": int((deltas > 0).sum()),
    "ties": int((deltas == 0).sum()),
    "losses": int((deltas < 0).sum()),
}
display(pd.DataFrame([asdict(tail_test), asdict(overall_test)], index=["tail", "overall"]))
print(direction)

# %% [markdown]
# ## 5. Материализуем второй submission и валидируем контракт
#
# Для публичной попытки сохраняем конфигурацию, выбранную на dev-tail, даже если
# малый test-tail даёт широкий CI. Статтест отвечает на вопрос о доказанном
# улучшении внутреннего baseline; публичная попытка дополнительно отвечает на
# вопрос о переносе при обнаруженном shift. Первый LTR submission берётся из
# `answer_finetuned_dense_ltr.csv` — явного выхода notebook 13 — и копируется в
# `submissions/attempt_1_finetuned_ltr.csv`; новый — в `attempt_2_robust_rrf.csv`
# и в корневой `answer.csv`. Корневой `answer.csv` для этого не используется:
# на чистом checkout в нём может лежать результат любого предыдущего notebook.

# %%
benchmark_channels = build_channels(rankings[rankings.split.eq("benchmark")])
benchmark_predictions = fuse_channels(
    benchmark_channels, WEIGHT_CONFIGS[selected_name], top_k=50
)

submission_dir = ROOT / "submissions"
submission_dir.mkdir(exist_ok=True)
attempt_1 = submission_dir / "attempt_1_finetuned_ltr.csv"
attempt_1_source = ROOT / "answer_finetuned_dense_ltr.csv"  # output of notebook 13
if not attempt_1.exists() and attempt_1_source.exists():
    shutil.copy2(attempt_1_source, attempt_1)
elif not attempt_1_source.exists():
    print("notebook 13 output not found; attempt 1 is not archived")

answer = pd.DataFrame({
    "query_id": benchmark_queries.query_id.astype(str),
    "answer": [
        " ".join(benchmark_predictions[str(query_id)])
        for query_id in benchmark_queries.query_id.astype(str)
    ],
})

corpus_ids = set(benchmark_items.item_id.astype(str))
assert list(answer.columns) == ["query_id", "answer"]
assert len(answer) == len(benchmark_queries) == answer.query_id.nunique()
assert set(answer.query_id) == set(benchmark_queries.query_id.astype(str))
assert answer.query_id.str.fullmatch(r"[A-Za-z0-9]{16}").all()
for item_string in answer.answer:
    item_ids = item_string.split()
    assert 1 <= len(item_ids) <= 50
    assert len(item_ids) == len(set(item_ids))
    assert all(re.fullmatch(r"[0-9a-f]{16}", item_id) for item_id in item_ids)
    assert set(item_ids).issubset(corpus_ids)

attempt_2 = submission_dir / "attempt_2_robust_rrf.csv"
answer.to_csv(attempt_2, index=False)
answer.to_csv(ROOT / "answer.csv", index=False)

metrics = {
    "public_attempt_1_recall": 0.698370,
    "shift": shift,
    "tail_definition": f"train normalized search_query frequency <= {TAIL_MAX_FREQUENCY}",
    "dev_tail_queries": len(dev_tail_ids),
    "test_tail_queries": len(test_tail_ids),
    "rrf_k": RRF_K,
    "weight_configs": {key: list(value) for key, value in WEIGHT_CONFIGS.items()},
    "selected_on_dev_tail": selected_name,
    "selected_weights": list(WEIGHT_CONFIGS[selected_name]),
    "comparison": comparison.to_dict("records"),
    "bonferroni_confidence": confidence,
    "selected_vs_bm25_fine_tail_test": asdict(tail_test),
    "selected_vs_bm25_fine_overall_test": asdict(overall_test),
    "test_tail_directions": direction,
    "attempt_2_rows": len(answer),
}
(ROOT / "reports/distribution_shift_rrf_metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
)
print({
    "submission": str(attempt_2),
    "rows": len(answer),
    "items_per_query_min": int(answer.answer.str.split().str.len().min()),
    "items_per_query_max": int(answer.answer.str.split().str.len().max()),
    "selected": selected_name,
})

# %% [markdown]
# ## Интерпретация
#
# Если selected method проходит строгий tail-gate, у нас есть внутреннее
# подтверждение робастности. Если CI пересекает ноль, это не превращается в
# заявление о доказанном приросте: submission остаётся контролируемой внешней
# проверкой гипотезы, мотивированной большим измеренным shift. Следующий шаг
# зависит от публичного score: он позволит различить ошибку LTR/generalization
# и более фундаментальный mismatch retrieval corpus/query distribution.
