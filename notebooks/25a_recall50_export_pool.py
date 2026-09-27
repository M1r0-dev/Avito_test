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
# # 25A. Экспорт пула C2 top-200 для GPU-подбора Recall@50 objective
#
# Notebook 24 зафиксировал пул attempt 6: `BM25 + zero-shot BGE-M3 + LoRA v2`,
# union top-100 каналов, обрезанный до top-200 по RRF-признаку `fused`.
# Selector в нём обучался `YetiRankPairwise`, который не знает про cutoff 50.
#
# Stage 25B на Kaggle (2×T4) перебирает objective и параметры CatBoost. Этот
# notebook только готовит для него неизменённый вход, поэтому признаки, split и
# baseline берутся строго из notebook 19 (запускается как prerequisite, как в
# notebook 24). Labels экспортируются полностью, включая positives вне пула:
# знаменатель macro Recall@50 должен совпадать с offline-оценкой репозитория.
#
# Выходы (`artifacts/recall50_kaggle_input/`, не коммитятся, загружаются как
# private Kaggle dataset):
#
# - `c2_validation.parquet`, `c2_benchmark.parquet` — признаки пула;
# - `relevant.parquet` — все relevant пары holdout;
# - `splits.json` — dev/test/tail и 2-fold cross-fitting на dev;
# - `attempt4_test_top50.parquet` — test-ранжирование attempt 4 (C3) для
#   локального статистического gate в 25C; на Kaggle не используется.

# %%
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import pandas as pd

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent

# Воспроизводим признаки и модели attempt 4 тем же кодом, что notebook 24.
get_ipython().run_line_magic("run", str(ROOT / "notebooks/19_lora_v2_evaluation.py"))

TOP_POOL = 200  # как в notebook 24: буквальное сжатие 200 → 50
OUT = ROOT / "artifacts/recall50_kaggle_input"
OUT.mkdir(parents=True, exist_ok=True)
C2_FEATURES = feature_columns("C2")

# %% [markdown]
# ## 1. Validation и benchmark пулы
#
# Ограничение `fused <= 200` не использует labels и одинаково применяется к
# holdout и benchmark. Порядок строк внутри запроса — по `fused`, чтобы
# tie-break на Kaggle совпадал с notebook 24.

# %%
def top_pool(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[frame.fused.le(TOP_POOL)]
    return frame.sort_values(["query_key", "fused"], kind="stable").reset_index(drop=True)


c2_validation = top_pool(features_by_system["C2"])
c2_benchmark = top_pool(pd.concat([
    system_features("C2", ("benchmark", str(query_id)))
    for query_id in benchmark_queries.query_id
], ignore_index=True))
c2_validation[["query_key", "item_id", "label", *C2_FEATURES]].to_parquet(
    OUT / "c2_validation.parquet", index=False)
c2_benchmark[["query_key", "item_id", *C2_FEATURES]].to_parquet(
    OUT / "c2_benchmark.parquet", index=False)
print("validation", c2_validation.shape, "benchmark", c2_benchmark.shape)

# %% [markdown]
# ## 2. Labels, split и test-ранжирование attempt 4

# %%
labels[["eval_query_id", "item_id"]].astype(str).to_parquet(OUT / "relevant.parquet", index=False)
splits = {
    "features": C2_FEATURES,
    "dev": sorted(dev_ids), "test": sorted(test_ids), "tail": sorted(tail_ids),
    "cross_folds": [{"fit": sorted(fit), "held": sorted(held)} for fit, held in cross_folds],
    "benchmark_query_ids": [str(q) for q in benchmark_queries.query_id],
}
(OUT / "splits.json").write_text(json.dumps(splits), encoding="utf-8")

attempt4 = pd.DataFrame([
    {"query_id": query_id, "rank": rank, "item_id": item_id}
    for query_id, items_ in test_predictions["C3"].items()
    for rank, item_id in enumerate(items_, 1)
])
attempt4.to_parquet(OUT / "attempt4_test_top50.parquet", index=False)
(OUT / "dataset-metadata.json").write_text(json.dumps({
    "title": "Avito Recall50 Selector Pool",
    "id": "m1r0tvorxc/avito-recall50-selector-pool",
    "licenses": [{"name": "other"}],
}, indent=2), encoding="utf-8")

for path in sorted(OUT.glob("*.parquet")) + [OUT / "splits.json"]:
    print(path.name, hashlib.sha256(path.read_bytes()).hexdigest()[:16])
