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
# # 27. Online CPU dense search не должен ронять Recall@50
#
# Все оценки финального решения (attempt 6) получены по dense rankings,
# которые Kaggle считал на T4 в **fp16**. Online-путь под latency guardrail
# работает на CPU (`avito_retrieval.dense_search.ExactDenseIndex`) в **fp32**,
# а local-канал берёт скоры из того же произведения, что и global. Близкие
# скоры могут упорядочиться иначе, поэтому перед замером latency проверяется,
# что online-реализация — это та же система по качеству.
#
# Проверяется канал LoRA v2: для него сохранены FP16 passage/query vectors
# (stage 10B). Zero-shot использует тот же код поиска, его векторы не
# сохранялись. Протокол:
#
# 1. CPU-rankings v2 для всех 4 904 запросов из сохранённых query vectors;
#    совпадение с GPU-rankings Kaggle.
# 2. Признаки selector пересобираются с CPU-rankings v2 (всё прочее то же),
#    три модели attempt 6 (`models/pu_selector_seed*.cbm`) не переобучаются.
# 3. **Non-inferiority** на test-tail и test против offline attempt 6:
#    граница `−0.005` Recall@50 (половина наименьшего эффекта, который мы
#    принимали), одностороннее `alpha=.025` на endpoint — нижняя граница 97.5%
#    bootstrap CI должна быть выше `−0.005`.

# %%
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRanker
from sklearn.model_selection import StratifiedKFold

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from avito_retrieval.dense_search import ExactDenseIndex
from avito_retrieval.filters import requested_min_rating
from avito_retrieval.learned_fusion import ITEM_COLUMNS, parse_rank
from avito_retrieval.metrics import recall_at_k
from avito_retrieval.pu_selector import (
    PU_SEEDS, V2_LOCAL_WEIGHT, build_features, channel_lists, ensemble_top50, load_channels,
)
from avito_retrieval.statistics import paired_recall_test, per_query_recall

pd.set_option("display.width", 200)
V2_DIR = ROOT / "artifacts/finetuned_v2_kaggle"
MARGIN = -0.005
ALPHA = 0.025
RANDOM_SEED = 42

# %% [markdown]
# ## 1. CPU-rankings v2 и их совпадение с GPU

# %%
items = pd.read_parquet(ROOT / "artifacts/bm25/items.parquet")
assert (pd.read_parquet(V2_DIR / "item_order.parquet").item_id.astype(str).values
        == items.item_id.astype(str).values).all()
index = ExactDenseIndex(
    np.load(V2_DIR / "passage_embeddings_v2_fp16.npy"), np.load(V2_DIR / "passage_item_rows.npy"),
    items.item_id.astype(str).to_numpy(), items.item_location_id.to_numpy(),
    items.item_category_id.to_numpy(), items.item_rating.fillna(-1).to_numpy(), V2_LOCAL_WEIGHT,
)
query_order = pd.read_parquet(V2_DIR / "query_order.parquet")
query_vectors = np.load(V2_DIR / "query_embeddings_v2_fp16.npy").astype(np.float32)

manifest = pd.read_parquet(ROOT / "artifacts/validation/manifest.parquet")
benchmark_queries = pd.read_parquet(ROOT / "dataset/benchmark_queries.parquet")
query_rows = pd.concat([
    manifest.assign(query_key=manifest.eval_query_id, split="validation"),
    benchmark_queries.assign(query_key=benchmark_queries.query_id, split="benchmark"),
], ignore_index=True).set_index(["split", "query_key"])

cpu_v2: dict[tuple[str, str], tuple[list[str], list[str]]] = {}
for start in range(0, len(query_order), 64):
    batch = query_order.iloc[start:start + 64]
    scores = query_vectors[start:start + 64] @ index.matrix.T
    for offset, key in enumerate(zip(batch.split, batch.query_key.astype(str))):
        query = query_rows.loc[key]
        cpu_v2[key] = index.lists_from_scores(
            scores[offset], int(query.search_category),
            requested_min_rating(query.search_infm_params_text), int(query.search_location_id))

def overlap(cpu_list: list[str], gpu_list: list[str], depth: int) -> float:
    # Два пустых списка (в локации запроса нет объявлений, ~17% holdout) —
    # полное совпадение, а не 0.
    a, b = set(cpu_list[:depth]), set(gpu_list[:depth])
    return 1.0 if not a and not b else len(a & b) / max(len(a), len(b))


gpu = pd.read_parquet(V2_DIR / "finetuned_v2_dense_rankings.parquet")
agreement = []
for row in gpu.itertuples(index=False):
    cpu_global, cpu_local = cpu_v2[(row.split, str(row.query_key))]
    for name, cpu_list, gpu_list in (("global", cpu_global, parse_rank(row.finetuned_v2_global)),
                                     ("local", cpu_local, parse_rank(row.finetuned_v2_local))):
        agreement.append({
            "channel": name, "identical": cpu_list == gpu_list,
            "empty": not gpu_list,
            "overlap_at_50": overlap(cpu_list, gpu_list, 50),
            "overlap_at_250": overlap(cpu_list, gpu_list, 250),
        })
agreement = pd.DataFrame(agreement).groupby("channel").mean()
display(agreement.round(5))

# %% [markdown]
# ## 2. Selector attempt 6 на CPU-rankings v2 против offline attempt 6

# %%
# Notebook 19: fold 0 (dev) = test-часть первого разбиения, test — всё остальное.
folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_SEED)
dev_index = next(iter(folds.split(manifest, manifest.primary_stratum)))[1]
test = manifest.drop(index=dev_index).reset_index(drop=True)
test_ids = set(test.eval_query_id.astype(str))
tail_ids = set(manifest.loc[manifest.query_frequency.le(1), "eval_query_id"].astype(str))
splits_export = json.loads((ROOT / "artifacts/recall50_kaggle_input/splits.json").read_text())
assert test_ids == set(splits_export["test"]), "test split must be the notebook-19 test half"

item_frame = pd.read_parquet(ROOT / "dataset/benchmark_items.parquet", columns=ITEM_COLUMNS)
models = []
for seed in PU_SEEDS:
    model = CatBoostRanker()
    model.load_model(str(ROOT / f"models/pu_selector_seed{seed}.cbm"))
    models.append(model)

offline_channels = load_channels(ROOT, "validation")
online_channels = {}
for key, lists in offline_channels.items():
    cpu_global, cpu_local = cpu_v2[("validation", key)]
    online_channels[key] = channel_lists(lists["bm25"], lists["bm25_plain"], lists["zero_global"],
                                         lists["zero_local"], cpu_global, cpu_local)

offline = ensemble_top50(build_features(test, "eval_query_id", offline_channels, item_frame), models)
online = ensemble_top50(build_features(test, "eval_query_id", online_channels, item_frame), models)
labels = pd.read_parquet(ROOT / "artifacts/validation/labels.parquet")
relevant = labels.groupby("eval_query_id").item_id.agg(set).to_dict()

attempt4 = pd.read_parquet(ROOT / "artifacts/recall50_kaggle_input/attempt4_test_top50.parquet")
attempt4 = attempt4.sort_values(["query_id", "rank"]).groupby("query_id").item_id.agg(list).to_dict()
print("offline attempt 6 test Recall@50:", recall_at_k(offline, {q: relevant[q] for q in test_ids}),
      "| attempt 4:", recall_at_k(attempt4, {q: relevant[q] for q in test_ids}))

rows = []
for segment, ids in {"test_tail": test_ids & tail_ids, "test": test_ids}.items():
    result = paired_recall_test(online, offline, relevant, query_ids=sorted(ids), k=50,
                                confidence=1 - 2 * ALPHA, n_resamples=20_000, seed=RANDOM_SEED)
    new = per_query_recall(online, relevant, query_ids=sorted(ids))
    old = per_query_recall(offline, relevant, query_ids=sorted(ids))
    differences = np.asarray([new[q] - old[q] for q in sorted(ids)])
    rows.append({
        "segment": segment,
        "online_cpu": recall_at_k({q: online[q] for q in ids}, {q: relevant[q] for q in ids}),
        "offline_gpu": recall_at_k({q: offline[q] for q in ids}, {q: relevant[q] for q in ids}),
        **asdict(result),
        "wins": int((differences > 0).sum()), "losses": int((differences < 0).sum()),
        "identical_top50": float(np.mean([set(online[q]) == set(offline[q]) for q in ids])),
        "non_inferior": bool(result.ci_low > MARGIN),
    })
non_inferiority = pd.DataFrame(rows)
display(non_inferiority.round(6))

# %% [markdown]
# ## 3. Benchmark: насколько online CPU-ответ совпадает с attempt 6

# %%
benchmark_offline = load_channels(ROOT, "benchmark")
benchmark_online = {}
for key, lists in benchmark_offline.items():
    cpu_global, cpu_local = cpu_v2[("benchmark", key)]
    benchmark_online[key] = channel_lists(lists["bm25"], lists["bm25_plain"], lists["zero_global"],
                                          lists["zero_local"], cpu_global, cpu_local)
online_benchmark = ensemble_top50(
    build_features(benchmark_queries, "query_id", benchmark_online, item_frame), models)
attempt6 = pd.read_csv(ROOT / "submissions/attempt_6_pu_without_lora_v1.csv", dtype=str).set_index("query_id").answer
benchmark_overlap = float(np.mean([
    len(set(online_benchmark[q]) & set(attempt6[q].split())) / 50 for q in attempt6.index]))
benchmark_identical = float(np.mean([set(online_benchmark[q]) == set(attempt6[q].split()) for q in attempt6.index]))
print({"benchmark_top50_overlap": benchmark_overlap, "benchmark_identical_sets": benchmark_identical})

report = {
    "channel_agreement_cpu_fp32_vs_gpu_fp16": agreement.reset_index().to_dict("records"),
    "non_inferiority_margin": MARGIN, "alpha_one_sided": ALPHA,
    "test": non_inferiority.to_dict("records"),
    "benchmark": {"top50_overlap_with_attempt6": benchmark_overlap,
                  "identical_top50_sets": benchmark_identical},
}
(ROOT / "reports/online_cpu_fidelity.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print("non-inferior on both endpoints:", bool(non_inferiority.non_inferior.all()))
