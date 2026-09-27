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
# # 28. ONNX Runtime encoder: та же система, в 3–5 раз быстрее
#
# Latency attempt 6 на ноутбуке (`reports/latency_final_cpu.json`): p95
# `872 ms` последовательно и `649 ms` с параллельными ветками при guardrail
# `500 ms`; из них по ~`215 ms` p50 на каждый из двух проходов 568M-encoder.
# `scripts/export_onnx_encoder.py` экспортирует тот же encoder в ONNX **без
# квантизации** (fp32, те же веса), а ONNX Runtime сливает attention, bias+GELU
# и skip+LayerNorm.
#
# Текст запроса для обоих dense-каналов — `dense_query_text`: Kaggle kernels
# USER-bge-m3 кодировали его без префикса `"query: "` (первый прогон этого
# notebook с префиксом дал косинус ~0.93 к сохранённым векторам v2 и падение
# Recall@50 — это была ошибка интеграции, а не ONNX).
#
# Критерий приемлемости — не «recall не упал статистически», а более сильный:
# **выход retrieval не меняется**. Проверяется на всех 4 904 запросах
# (validation + benchmark):
#
# 1. векторы ORT (online-режим, batch=1) против PyTorch SentenceTransformer
#    (`max_seq_length=256`, как в Kaggle kernels): косинус и max |Δ|;
# 2. item-rankings global/local по реальной матрице LoRA v2 (notebook 27) с
#    векторами PyTorch и ORT одного и того же encoder — доля идентичных.
#
# Если rankings идентичны, признаки и top-50 selector идентичны, и Recall@50
# совпадает на каждом запросе (paired delta ровно 0), без статистического
# допуска. Если нет (шум ~1e-6 переставляет почти равные скоры среди 3 000
# passages), решает раздел 3: **настоящий** LoRA v2 (merged-модель Kaggle
# stage 10A, `scripts/export_onnx_encoder.py --model ...`) через ONNX Runtime
# и CPU-поиск против offline attempt 6 — non-inferiority как в notebook 27
# (граница `−0.005`, одностороннее `alpha=.025` на endpoint).

# %%
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRanker
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedKFold

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from avito_retrieval.dense_search import ExactDenseIndex
from avito_retrieval.learned_fusion import ITEM_COLUMNS
from avito_retrieval.metrics import recall_at_k
from avito_retrieval.pu_selector import (
    PU_SEEDS, V2_LOCAL_WEIGHT, build_features, channel_lists, ensemble_top50, load_channels,
)
from avito_retrieval.statistics import paired_recall_test, per_query_recall
from avito_retrieval.filters import requested_min_rating
from avito_retrieval.onnx_encoder import MAX_LENGTH, OnnxQueryEncoder
from avito_retrieval.text import dense_query_text

MODEL = "deepvk/USER-bge-m3"
MODEL_REVISION = "0cc6cfe48e260fb0474c753087a69369e88709ae"
ONNX_DIR = ROOT / "artifacts/onnx/user_bge_m3"
V2_DIR = ROOT / "artifacts/finetuned_v2_kaggle"

# %% [markdown]
# ## 1. Векторы: PyTorch против ONNX Runtime

# %%
manifest = pd.read_parquet(ROOT / "artifacts/validation/manifest.parquet")
benchmark_queries = pd.read_parquet(ROOT / "dataset/benchmark_queries.parquet")
queries = pd.concat([manifest, benchmark_queries], ignore_index=True)
texts = [dense_query_text(row) for row in queries.to_dict("records")]

reference = SentenceTransformer(MODEL, revision=MODEL_REVISION, device="cpu")
reference.max_seq_length = MAX_LENGTH
with torch.inference_mode():
    torch_vectors = reference.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
del reference

encoder = OnnxQueryEncoder(ONNX_DIR, threads=4)
started = time.perf_counter()
onnx_vectors = np.stack([encoder.encode(text) for text in texts])
print(f"ONNX batch=1: {(time.perf_counter() - started) / len(texts) * 1e3:.1f} ms per query (mean)")

cosine = (torch_vectors * onnx_vectors).sum(axis=1)
vectors = {"queries": len(texts), "min_cosine": float(cosine.min()),
           "max_abs_diff": float(np.abs(torch_vectors - onnx_vectors).max())}
print(vectors)

# %% [markdown]
# ## 2. Item-rankings по реальной матрице LoRA v2

# %%
items = pd.read_parquet(ROOT / "artifacts/bm25/items.parquet")
index = ExactDenseIndex(
    np.load(V2_DIR / "passage_embeddings_v2_fp16.npy"), np.load(V2_DIR / "passage_item_rows.npy"),
    items.item_id.astype(str).to_numpy(), items.item_location_id.to_numpy(),
    items.item_category_id.to_numpy(), items.item_rating.fillna(-1).to_numpy(), V2_LOCAL_WEIGHT,
)
identical = {"global": 0, "local": 0}
for start in range(0, len(texts), 64):
    torch_scores = torch_vectors[start:start + 64] @ index.matrix.T
    onnx_scores = onnx_vectors[start:start + 64] @ index.matrix.T
    for offset, query in enumerate(queries.iloc[start:start + 64].itertuples(index=False)):
        filters = (int(query.search_category), requested_min_rating(query.search_infm_params_text),
                   int(query.search_location_id))
        a = index.lists_from_scores(torch_scores[offset], *filters)
        b = index.lists_from_scores(onnx_scores[offset], *filters)
        identical["global"] += a[0] == b[0]
        identical["local"] += a[1] == b[1]
rankings = {name: count / len(texts) for name, count in identical.items()}
print("identical item rankings:", rankings)

# %% [markdown]
# ## 3. Настоящий LoRA v2 через ONNX Runtime: весь online-канал против offline
#
# Reference здесь — то, что реально оценивалось: векторы и rankings Kaggle
# (T4, fp16). Online: ORT fp32 encoder (batch=1) + CPU fp32 поиск. Модели
# selector attempt 6 не переобучаются. Zero-shot канал остаётся offline: его
# ORT-векторы отличаются от PyTorch на ~1e-6 (раздел 1), что на три порядка
# меньше расхождения fp16-GPU против fp32-CPU, проверяемого здесь для v2.

# %%
V2_ONNX_DIR = ROOT / "artifacts/onnx/lora_v2"
MARGIN, ALPHA, RANDOM_SEED = -0.005, 0.025, 42
v2_encoder = OnnxQueryEncoder(V2_ONNX_DIR, threads=4)
query_order = pd.read_parquet(V2_DIR / "query_order.parquet")
rows_by_key = {("validation", str(r["eval_query_id"])): r for r in manifest.to_dict("records")}
rows_by_key.update({("benchmark", str(r["query_id"])): r for r in benchmark_queries.to_dict("records")})
order_keys = list(zip(query_order.split, query_order.query_key.astype(str)))
v2_onnx = np.stack([v2_encoder.encode(dense_query_text(rows_by_key[key])) for key in order_keys])
v2_kaggle = np.load(V2_DIR / "query_embeddings_v2_fp16.npy").astype(np.float32)
v2_cosine = (v2_onnx * v2_kaggle).sum(axis=1) / np.linalg.norm(v2_kaggle, axis=1)
v2_vectors = {"min_cosine_vs_kaggle_fp16": float(v2_cosine.min()),
              "mean_cosine_vs_kaggle_fp16": float(v2_cosine.mean())}
print(v2_vectors)

online_v2 = {}
for start in range(0, len(order_keys), 64):
    scores = v2_onnx[start:start + 64] @ index.matrix.T
    for offset, key in enumerate(order_keys[start:start + 64]):
        row = rows_by_key[key]
        online_v2[key] = index.lists_from_scores(
            scores[offset], int(row["search_category"]),
            requested_min_rating(row["search_infm_params_text"]), int(row["search_location_id"]))

folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_SEED)
dev_index = next(iter(folds.split(manifest, manifest.primary_stratum)))[1]
test = manifest.drop(index=dev_index).reset_index(drop=True)  # test half of notebook 19
test_ids = set(test.eval_query_id.astype(str))
tail_ids = set(manifest.loc[manifest.query_frequency.le(1), "eval_query_id"].astype(str))
item_frame = pd.read_parquet(ROOT / "dataset/benchmark_items.parquet", columns=ITEM_COLUMNS)
models = []
for seed in PU_SEEDS:
    model = CatBoostRanker()
    model.load_model(str(ROOT / f"models/pu_selector_seed{seed}.cbm"))
    models.append(model)


def with_online_v2(channels: dict, split: str) -> dict:
    return {key: channel_lists(lists["bm25"], lists["bm25_plain"], lists["zero_global"],
                               lists["zero_local"], *online_v2[(split, key)])
            for key, lists in channels.items()}


offline_channels = load_channels(ROOT, "validation")
offline = ensemble_top50(build_features(test, "eval_query_id", offline_channels, item_frame), models)
online = ensemble_top50(build_features(test, "eval_query_id", with_online_v2(offline_channels, "validation"),
                                       item_frame), models)
labels = pd.read_parquet(ROOT / "artifacts/validation/labels.parquet")
relevant = labels.groupby("eval_query_id").item_id.agg(set).to_dict()

rows = []
for segment, ids in {"test_tail": test_ids & tail_ids, "test": test_ids}.items():
    result = paired_recall_test(online, offline, relevant, query_ids=sorted(ids), k=50,
                                confidence=1 - 2 * ALPHA, n_resamples=20_000, seed=RANDOM_SEED)
    new = per_query_recall(online, relevant, query_ids=sorted(ids))
    old = per_query_recall(offline, relevant, query_ids=sorted(ids))
    differences = np.asarray([new[q] - old[q] for q in sorted(ids)])
    rows.append({
        "segment": segment,
        "online_onnx_cpu": recall_at_k({q: online[q] for q in ids}, {q: relevant[q] for q in ids}),
        "offline_gpu": recall_at_k({q: offline[q] for q in ids}, {q: relevant[q] for q in ids}),
        **asdict(result),
        "wins": int((differences > 0).sum()), "losses": int((differences < 0).sum()),
        "non_inferior": bool(result.ci_low > MARGIN),
    })
non_inferiority = pd.DataFrame(rows)
display(non_inferiority.round(6))

benchmark_online = ensemble_top50(build_features(
    benchmark_queries, "query_id", with_online_v2(load_channels(ROOT, "benchmark"), "benchmark"), item_frame), models)
attempt6 = pd.read_csv(ROOT / "submissions/attempt_6_pu_without_lora_v1.csv", dtype=str).set_index("query_id").answer
benchmark_overlap = float(np.mean([
    len(set(benchmark_online[q]) & set(attempt6[q].split())) / 50 for q in attempt6.index]))
print({"benchmark_top50_overlap_with_attempt6": benchmark_overlap,
       "non_inferior_on_both_endpoints": bool(non_inferiority.non_inferior.all())})

report = {"encoder": MODEL, "revision": MODEL_REVISION, "max_length": MAX_LENGTH,
          "precision": "fp32, no quantization", "vectors": vectors,
          "identical_item_rankings_v2_matrix": rankings,
          "lora_v2_onnx": {**v2_vectors, "non_inferiority_margin": MARGIN, "alpha_one_sided": ALPHA,
                           "test": non_inferiority.to_dict("records"),
                           "benchmark_top50_overlap_with_attempt6": benchmark_overlap}}
(ROOT / "reports/onnx_encoder_equivalence.json").write_text(
    json.dumps(report, indent=2, default=float), encoding="utf-8")
