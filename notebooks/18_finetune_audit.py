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
# # Аудит LoRA v1: где он помогает и сколько данных не использовано
#
# Fine-tuned USER-bge-m3 (stage 8A/8B) — самый весомый канал попытки 2
# (вес 1.25). Перед его переобучением нужно понять две вещи:
#
# 1. **Где fine-tune даёт прирост.** Benchmark на 72% состоит из редких
#    запросов (train frequency `<=1`), holdout — на 15%. Если прирост живёт
#    только на head, holdout переоценивает канал для benchmark.
# 2. **Сколько данных v1 не видел.** 8A брал только пары с объявлениями из
#    benchmark-корпуса и по одной паре на объявление.
#
# Гипотеза «прирост — это запоминание текстов из обучающих пар» тоже
# проверяется: holdout исключал сигнатуры запросов, но не тексты.
#
# Сравнение каналов — dense-only (global/local RRF с весами notebooks 05/08),
# на всём holdout: оба канала не обучались на holdout-сигнатурах, и здесь
# ничего не выбирается, поэтому dev/test не разделяются. Для каждого сегмента —
# paired bootstrap 95% CI и one-sided randomization test.

# %%
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import sys
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from avito_retrieval.fusion import reciprocal_rank_fusion
from avito_retrieval.metrics import recall_at_k
from avito_retrieval.statistics import paired_recall_test

QUERY_COLUMNS = ["search_query", "search_location_id", "search_is_delivery_search",
                 "search_infm_params_text", "search_category"]
PAIRS_PATH = ROOT / "artifacts/finetuned_train_kaggle/finetune_pairs.parquet"


def clean(value: object) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return " ".join(str(value).replace("ё", "е").lower().split())


def parse_rank(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and value != value) or not str(value):
        return []
    return str(value).split()


# %% [markdown]
# ## 1. Обучающие пары v1
#
# Файл лежит в публичном output kernel 8A; при отсутствии скачивается
# анонимным endpoint Kaggle (как `scripts/fetch_public_kaggle_outputs.py`).

# %%
if not PAIRS_PATH.exists():
    PAIRS_PATH.parent.mkdir(parents=True, exist_ok=True)
    api = ("https://www.kaggle.com/api/v1/kernels/output?userName=m1r0tvorxc"
           "&kernelSlug=avito-user-bge-m3-domain-adaptation")
    with urllib.request.urlopen(api, timeout=60) as response:
        files = json.load(response)["files"]
    url = next(f.get("url") or f.get("urlNullable") for f in files
               if f["fileName"].endswith("finetune_pairs.parquet"))
    urllib.request.urlretrieve(url, PAIRS_PATH)
v1_pairs = pd.read_parquet(PAIRS_PATH)
v1_texts = set(v1_pairs.query_text)
print(f"v1 training pairs: {len(v1_pairs):,}")

# %% [markdown]
# ## 2. Fine-tuned против zero-shot по сегментам

# %%
manifest = pd.read_parquet(ROOT / "artifacts/validation/manifest.parquet").copy()
labels = pd.read_parquet(ROOT / "artifacts/validation/labels.parquet")
relevant = labels.groupby("eval_query_id").item_id.agg(set).to_dict()
manifest["query_text"] = [f"{clean(q)} {clean(f)}".strip()
                          for q, f in zip(manifest.search_query, manifest.search_infm_params_text)]
manifest["seen_in_v1_pairs"] = manifest.query_text.isin(v1_texts)
manifest["is_tail"] = manifest.query_frequency.le(1)

benchmark = pd.read_parquet(ROOT / "dataset/benchmark_queries.parquet")
benchmark_texts = [f"{clean(q)} {clean(f)}".strip()
                   for q, f in zip(benchmark.search_query, benchmark.search_infm_params_text)]
exposure = {
    "holdout_seen_share": float(manifest.seen_in_v1_pairs.mean()),
    "holdout_head_seen_share": float(manifest.loc[~manifest.is_tail, "seen_in_v1_pairs"].mean()),
    "holdout_tail_seen_share": float(manifest.loc[manifest.is_tail, "seen_in_v1_pairs"].mean()),
    "benchmark_seen_share": float(np.mean([text in v1_texts for text in benchmark_texts])),
}
display(exposure)

rankings = pd.read_parquet(ROOT / "artifacts/dense_kaggle/dense_rankings.parquet").merge(
    pd.read_parquet(ROOT / "artifacts/finetuned_dense_kaggle/finetuned_dense_rankings.parquet"),
    on=["query_key", "split"], validate="one_to_one")
rankings = rankings[rankings.split.eq("validation")]
zero, fine = {}, {}
for row in rankings.itertuples(index=False):
    zero[row.query_key] = reciprocal_rank_fusion(
        [parse_rank(row.dense_global), parse_rank(row.dense_local)], weights=[1.0, 1.15], rrf_k=60, top_k=250)
    fine[row.query_key] = reciprocal_rank_fusion(
        [parse_rank(row.finetuned_global), parse_rank(row.finetuned_local)], weights=[1.0, 1.5], rrf_k=60, top_k=250)

segments = {
    "all": manifest,
    "tail (freq<=1)": manifest[manifest.is_tail],
    "head (freq>1)": manifest[~manifest.is_tail],
    "text seen in v1 pairs": manifest[manifest.seen_in_v1_pairs],
    "text unseen": manifest[~manifest.seen_in_v1_pairs],
}
rows = []
for name, frame in segments.items():
    ids = sorted(frame.eval_query_id)
    for k in (50, 250):
        test = paired_recall_test({q: fine[q][:k] for q in ids}, {q: zero[q][:k] for q in ids},
                                  relevant, query_ids=ids, k=k, n_resamples=20_000, seed=42)
        rows.append({"segment": name, "n": len(ids), "k": k,
                     "zero": recall_at_k({q: zero[q] for q in ids}, {q: relevant[q] for q in ids}, k=k),
                     "fine": recall_at_k({q: fine[q] for q in ids}, {q: relevant[q] for q in ids}, k=k),
                     **asdict(test)})
segment_table = pd.DataFrame(rows)
display(segment_table.round(4))

# %% [markdown]
# **Вывод.** Прирост одинаков для текстов, встречавшихся и не встречавшихся в
# обучающих парах, то есть это не запоминание. Он сосредоточен на head:
# на хвосте прирост статистически не отличается от нуля. Канал учит частые
# типы запросов, а benchmark в основном из редких.

# %% [markdown]
# ## 3. Сколько пар доступно для обучения

# %%
train = pd.read_parquet(ROOT / "dataset/train.parquet",
                        columns=QUERY_COLUMNS + ["item_id", "item_title_raw", "item_description_raw"])
marked = train.merge(manifest[QUERY_COLUMNS].drop_duplicates().assign(_holdout=1),
                     on=QUERY_COLUMNS, how="left")
usable = marked[marked._holdout.isna()].copy()
usable["query_text"] = [f"{clean(q)} {clean(f)}".strip()
                        for q, f in zip(usable.search_query, usable.search_infm_params_text)]
corpus_ids = set(pd.read_parquet(ROOT / "dataset/benchmark_items.parquet", columns=["item_id"]).item_id)
pairs = usable.drop_duplicates(["query_text", "item_id"])
per_text = pairs.groupby("query_text").size()
data_headroom = {
    "train_rows_after_holdout_exclusion": len(usable),
    "item_text_available_share": float(usable.item_title_raw.notna().mean()),
    "unique_query_item_pairs": len(pairs),
    "unique_query_texts": int(per_text.size),
    "unique_items": int(pairs.item_id.nunique()),
    "pairs_with_item_in_benchmark_corpus_share": float(pairs.item_id.isin(corpus_ids).mean()),
    "v1_pairs": len(v1_pairs),
    "v1_share_of_available": len(v1_pairs) / len(pairs),
    "pairs_kept_if_capped_at_32_per_text": int(per_text.clip(upper=32).sum()),
}
display(data_headroom)

(ROOT / "reports/finetune_audit_metrics.json").write_text(json.dumps({
    "exposure": exposure, "segments": segment_table.to_dict("records"),
    "data_headroom": data_headroom,
}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print("saved reports/finetune_audit_metrics.json")
