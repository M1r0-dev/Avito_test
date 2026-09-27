# %% [markdown]
# # Stage 9 — zero-shot cross-encoder rerank of the attempt-2 pool (2×T4)
#
# Гипотеза (notebook 15): 75.8% промахов попытки 2 — selection misses. Позитив
# уже лежит в пуле BM25 + zero-shot dense + fine-tuned dense (медианный лучший
# ранг 67), но RRF не поднимает его в top-50. Cross-encoder читает запрос и
# объявление совместно и может лучше отобрать 50 из ~200 кандидатов.
#
# Почему именно так:
#
# - `BAAI/bge-reranker-v2-m3` — многоязычный reranker (Apache-2.0) на базе
#   BGE-M3, того же семейства, что USER-bge-m3; русский поддерживается;
# - **zero-shot**: модель не обучается на кликах train, поэтому не может
#   переобучиться на head-запросы — именно это сломало LTR на benchmark
#   (offline 0.860 → public 0.698);
# - скорится весь пул глубины 100 для validation и benchmark; локальный
#   notebook 16 выбирает глубину и вес fusion только на dev-tail;
# - модель скачивается один раз в главном процессе с закреплённой ревизией,
#   затем два процесса читают её локально, каждый на своём GPU (урок зависания
#   параллельной загрузки checkpoint в stage 8A).
#
# Вход: исходные parquet (приватный dataset `avito-candidate-data`) и
# `rerank_pool.parquet` (приватный dataset `avito-rerank-pool`, строится
# `scripts/build_rerank_pool.py`). Выход: `rerank_scores.parquet` с одним
# logit на пару (query_key, item_id) и `rerank_run.json`.

# %%
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import snapshot_download

MODEL = "BAAI/bge-reranker-v2-m3"
MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
MODEL_LICENSE = "apache-2.0"
# Запрос занимает ~10–60 токенов; остаток — заголовок, начало параметров и
# описание. 320 токенов покрывают заголовок и основную часть описания без
# квадратичного роста стоимости attention на длинных параметрах.
MAX_LENGTH = 320
# Параметры объявления бывают тысячами слов (EDA). Их начало содержит
# `Вид услуги`/`Тип услуги`, но часто и адрес, расписание, тип стоимости.
# Поэтому они идут последними: при усечении до MAX_LENGTH первым
# отрезается наименее информативный хвост, а не описание.
PARAMS_CHARS = 300
BATCH_SIZE = 64
OUTPUT = Path("/kaggle/working")


def locate(name: str) -> Path:
    """Find an input file regardless of the dataset mount layout."""
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one {name}, found {matches}")
    return matches[0]


def document_text(row: object) -> str:
    """Pre-registered item text: title, description, head of parameters.

    The order was fixed after a smoke test showed that a parameters-first
    layout spends a large part of the 320-token budget on address and
    schedule fields before the description starts.
    """
    parts = [
        str(row.item_title_raw or "").strip(),
        str(row.item_description_raw or "").strip(),
        str(row.item_infm_params_text or "")[:PARAMS_CHARS].strip(),
    ]
    return ". ".join(" ".join(part.split()) for part in parts if part)


# %% [markdown]
# ## Worker: one process per GPU
#
# Каждый процесс получает свою половину пар, сортирует её по длине документа
# (меньше padding в batch) и пишет logit'ы в отдельный shard.

# %%
WORKER = r'''
import sys, time
import numpy as np, pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path, pairs_path, output_path, max_length, batch_size = sys.argv[1:6]
max_length, batch_size = int(max_length), int(batch_size)
pairs = pd.read_parquet(pairs_path)
order = np.argsort(pairs.document.str.len().to_numpy())
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path, torch_dtype=torch.float16
).cuda().eval()
scores = np.empty(len(pairs), dtype=np.float32)
queries, documents = pairs["query"].to_numpy(), pairs["document"].to_numpy()
started = time.time()
with torch.inference_mode():
    for start in range(0, len(order), batch_size):
        index = order[start:start + batch_size]
        batch = tokenizer(
            list(queries[index]), list(documents[index]), padding=True,
            truncation="only_second", max_length=max_length, return_tensors="pt",
        ).to("cuda")
        scores[index] = model(**batch).logits.view(-1).float().cpu().numpy()
        if start // batch_size % 500 == 0:
            done = start + len(index)
            print(f"{done}/{len(order)} pairs, {done / (time.time() - started):.0f} pairs/s", flush=True)
pairs[["query_key", "split", "item_id"]].assign(score=scores).to_parquet(output_path, index=False)
print("worker done", len(order), flush=True)
'''

# %%
def main() -> None:
    gpus = torch.cuda.device_count()
    if gpus != 2:
        raise RuntimeError(f"Expected two GPUs, found {gpus}")
    model_path = snapshot_download(MODEL, revision=MODEL_REVISION)

    items = pd.read_parquet(
        locate("benchmark_items.parquet"),
        columns=["item_id", "item_title_raw", "item_infm_params_text", "item_description_raw"],
    )
    documents = dict(zip(items.item_id.astype(str), [document_text(row) for row in items.itertuples(index=False)]))
    pool = pd.read_parquet(locate("rerank_pool.parquet"))
    pairs = pool.assign(item_id=pool.candidates.str.split()).explode("item_id")
    pairs = pairs.rename(columns={"query_text": "query"})[["query_key", "split", "query", "item_id"]]
    pairs["document"] = pairs.item_id.map(documents)
    assert pairs.document.notna().all(), "pool contains items outside the corpus"
    print(f"queries={len(pool):,}; pairs={len(pairs):,}", flush=True)

    # Interleaved split keeps both shards balanced in query length and split.
    started = time.time()
    workers, shard_paths = [], []
    (OUTPUT / "rerank_worker.py").write_text(WORKER, encoding="utf-8")
    for shard in range(2):
        pairs_path = OUTPUT / f"pairs_{shard}.parquet"
        shard_path = OUTPUT / f"scores_{shard}.parquet"
        pairs.iloc[shard::2].to_parquet(pairs_path, index=False)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(shard)}
        workers.append(subprocess.Popen([
            sys.executable, str(OUTPUT / "rerank_worker.py"), model_path,
            str(pairs_path), str(shard_path), str(MAX_LENGTH), str(BATCH_SIZE),
        ], env=env))
        shard_paths.append(shard_path)
    codes = [worker.wait() for worker in workers]
    if any(codes):
        raise RuntimeError(f"worker exit codes: {codes}")

    scores = pd.concat([pd.read_parquet(path) for path in shard_paths], ignore_index=True)
    assert len(scores) == len(pairs) and not scores.duplicated(["query_key", "split", "item_id"]).any()
    scores.to_parquet(OUTPUT / "rerank_scores.parquet", index=False)
    for path in [*shard_paths, *(OUTPUT / f"pairs_{shard}.parquet" for shard in range(2))]:
        path.unlink()
    (OUTPUT / "rerank_run.json").write_text(json.dumps({
        "stage": "zero_shot_cross_encoder_rerank",
        "model": MODEL, "revision": MODEL_REVISION, "license": MODEL_LICENSE,
        "max_length": MAX_LENGTH, "params_chars": PARAMS_CHARS, "batch_size": BATCH_SIZE,
        "queries": int(len(pool)), "pairs": int(len(pairs)),
        "elapsed_seconds": round(time.time() - started, 1),
        "gpus": [torch.cuda.get_device_name(index) for index in range(gpus)],
    }, indent=2), encoding="utf-8")
    print("saved rerank_scores.parquet", flush=True)


# %%
main()
