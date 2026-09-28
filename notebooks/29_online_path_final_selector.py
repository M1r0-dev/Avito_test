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
# # 29. Online-путь с финальным selector (attempt 7) не хуже offline
#
# Notebooks 27–28 проверили online-путь (ONNX Runtime encoder + точный CPU
# поиск) с моделями attempt 6. Финальное решение — attempt 7: те же каналы и
# пул, но PU bags обучены Recall@50 lambda objective. Здесь тот же протокол
# повторяется с финальными моделями (`models/recall50_seed*_round*.cbm`):
#
# - LoRA v2 кодируется настоящей merged-моделью через ORT (`dense_query_text`,
#   batch=1) и ищется точно на CPU; BM25 и zero-shot — сохранённые rankings;
# - non-inferiority против offline attempt 7 на test-tail и test: граница
#   `−0.005`, одностороннее `alpha=.025` (нижняя граница 97.5% bootstrap CI);
# - на benchmark — совпадение top-50 с отправленным `answer.csv`.

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
sys.path.insert(0, str(ROOT / "scripts"))

from avito_retrieval.dense_search import ExactDenseIndex
from avito_retrieval.filters import requested_min_rating
from avito_retrieval.learned_fusion import ITEM_COLUMNS
from avito_retrieval.metrics import recall_at_k
from avito_retrieval.onnx_encoder import OnnxQueryEncoder
from avito_retrieval.pu_selector import V2_LOCAL_WEIGHT, build_features, channel_lists, ensemble_top50, load_channels
from avito_retrieval.statistics import paired_recall_test, per_query_recall
from avito_retrieval.text import dense_query_text
from generate_answer import load_models

pd.set_option("display.width", 200)
V2_DIR = ROOT / "artifacts/finetuned_v2_kaggle"
MARGIN, ALPHA, RANDOM_SEED = -0.005, 0.025, 42

# %% [markdown]
# ## 1. Online LoRA v2: ORT encoder + точный CPU-поиск

# %%
items = pd.read_parquet(ROOT / "artifacts/bm25/items.parquet")
index = ExactDenseIndex(
    np.load(V2_DIR / "passage_embeddings_v2_fp16.npy"), np.load(V2_DIR / "passage_item_rows.npy"),
    items.item_id.astype(str).to_numpy(), items.item_location_id.to_numpy(),
    items.item_category_id.to_numpy(), items.item_rating.fillna(-1).to_numpy(), V2_LOCAL_WEIGHT,
)
manifest = pd.read_parquet(ROOT / "artifacts/validation/manifest.parquet")
benchmark_queries = pd.read_parquet(ROOT / "dataset/benchmark_queries.parquet")
rows_by_key = {("validation", str(r["eval_query_id"])): r for r in manifest.to_dict("records")}
rows_by_key.update({("benchmark", str(r["query_id"])): r for r in benchmark_queries.to_dict("records")})
keys = list(rows_by_key)

encoder = OnnxQueryEncoder(ROOT / "artifacts/onnx/lora_v2", threads=4)
vectors = np.stack([encoder.encode(dense_query_text(rows_by_key[key])) for key in keys])
online_v2 = {}
for start in range(0, len(keys), 64):
    scores = vectors[start:start + 64] @ index.matrix.T
    for offset, key in enumerate(keys[start:start + 64]):
        row = rows_by_key[key]
        online_v2[key] = index.lists_from_scores(
            scores[offset], int(row["search_category"]),
            requested_min_rating(row["search_infm_params_text"]), int(row["search_location_id"]))

# %% [markdown]
# ## 2. Non-inferiority на test с финальными моделями

# %%
folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_SEED)
dev_index = next(iter(folds.split(manifest, manifest.primary_stratum)))[1]
test = manifest.drop(index=dev_index).reset_index(drop=True)  # test half of notebook 19
test_ids = set(test.eval_query_id.astype(str))
tail_ids = set(manifest.loc[manifest.query_frequency.le(1), "eval_query_id"].astype(str))
item_frame = pd.read_parquet(ROOT / "dataset/benchmark_items.parquet", columns=ITEM_COLUMNS)
models = load_models(ROOT / "models", "recall50")


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
print("offline attempt 7 test Recall@50:", recall_at_k(offline, {q: relevant[q] for q in test_ids}))

rows = []
for segment, ids in {"test_tail": test_ids & tail_ids, "test": test_ids}.items():
    result = paired_recall_test(online, offline, relevant, query_ids=sorted(ids), k=50,
                                confidence=1 - 2 * ALPHA, n_resamples=20_000, seed=RANDOM_SEED)
    new = per_query_recall(online, relevant, query_ids=sorted(ids))
    old = per_query_recall(offline, relevant, query_ids=sorted(ids))
    differences = np.asarray([new[q] - old[q] for q in sorted(ids)])
    rows.append({
        "segment": segment,
        "online": recall_at_k({q: online[q] for q in ids}, {q: relevant[q] for q in ids}),
        "offline": recall_at_k({q: offline[q] for q in ids}, {q: relevant[q] for q in ids}),
        **asdict(result), "wins": int((differences > 0).sum()), "losses": int((differences < 0).sum()),
        "non_inferior": bool(result.ci_low > MARGIN),
    })
non_inferiority = pd.DataFrame(rows)
display(non_inferiority.round(6))

# %% [markdown]
# ## 3. Benchmark: online-ответ против отправленного attempt 7

# %%
online_benchmark = ensemble_top50(build_features(
    benchmark_queries, "query_id", with_online_v2(load_channels(ROOT, "benchmark"), "benchmark"), item_frame), models)
submitted = pd.read_csv(ROOT / "answer.csv", dtype=str).set_index("query_id").answer
overlap = float(np.mean([len(set(online_benchmark[q]) & set(submitted[q].split())) / 50 for q in submitted.index]))
print({"benchmark_top50_overlap_with_answer_csv": overlap,
       "non_inferior_on_both_endpoints": bool(non_inferiority.non_inferior.all())})

(ROOT / "reports/online_final_selector.json").write_text(json.dumps({
    "selector": "attempt 7 (Recall@50 lambda objective, models/recall50_seed*_round*.cbm)",
    "non_inferiority_margin": MARGIN, "alpha_one_sided": ALPHA,
    "test": non_inferiority.to_dict("records"),
    "benchmark_top50_overlap_with_answer_csv": overlap,
}, indent=2, default=float), encoding="utf-8")
