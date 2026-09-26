"""Kaggle 2×T4 job: encode fixed passages and retrieve dense candidates.

The job produces item-level rankings for the offline holdout and benchmark.
It intentionally exports rankings rather than a multi-gigabyte embedding matrix.
"""

from __future__ import annotations

import hashlib
import gc
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "deepvk/USER-bge-m3"
QUERY_COLUMNS = [
    "search_query", "search_location_id", "search_is_delivery_search",
    "search_infm_params_text", "search_category",
]
CHUNK_WORDS = 140
CHUNK_OVERLAP = 30
MAX_CHUNKS = 4
TOP_CHUNKS_PER_SHARD = 1500
TOP_ITEMS = 250
TOKEN_RE = re.compile(r"(?u)[\w]+(?:[-'][\w]+)*")
RATING_RE = re.compile(r"рейтинг[^\d]{0,20}([1-5](?:[.,]\d+)?)", re.IGNORECASE)


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).replace("ё", "е").lower().split())


def tokenize(value: object) -> list[str]:
    return TOKEN_RE.findall(clean(value))


def make_passages(items: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    passages: list[str] = []
    item_rows: list[int] = []
    step = CHUNK_WORDS - CHUNK_OVERLAP
    for item_row, row in enumerate(items.itertuples(index=False)):
        title = clean(row.item_title_raw)
        content = tokenize(row.item_infm_params_text) + ["описание"] + tokenize(row.item_description_raw)
        if not content:
            content = [""]
        for chunk_no, start in enumerate(range(0, len(content), step)):
            if chunk_no >= MAX_CHUNKS:
                break
            body = " ".join(content[start : start + CHUNK_WORDS])
            # USER-bge-m3 was trained without E5-style query/passage prefixes.
            passages.append(f"{title}. {body}".strip())
            item_rows.append(item_row)
    return passages, np.asarray(item_rows, dtype=np.int32)


def query_text(row: pd.Series) -> str:
    return f"{clean(row.search_query)} {clean(row.search_infm_params_text)}".strip()


def requested_min_rating(value: object) -> float | None:
    match = RATING_RE.search(str(value))
    return float(match.group(1).replace(",", ".")) if match else None


def stable_query_id(row: pd.Series) -> str:
    payload = "\x1f".join(str(row[c]) for c in QUERY_COLUMNS)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def encode_multi_gpu(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if len(devices) < 2:
        raise RuntimeError(f"This job expects 2 GPUs, found {devices}")
    # SentenceTransformers uses one worker per target device.
    pool = model.start_multi_process_pool(target_devices=devices)
    try:
        embeddings = model.encode_multi_process(
            texts,
            pool,
            batch_size=batch_size,
            chunk_size=4096,
            normalize_embeddings=True,
        )
    finally:
        model.stop_multi_process_pool(pool)
    return np.asarray(embeddings, dtype=np.float32)


def gpu_top_chunks(query_embeddings: np.ndarray, passage_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact inner-product search sharded over both T4 GPUs."""
    n_gpu = torch.cuda.device_count()
    shards = np.array_split(np.arange(len(passage_embeddings)), n_gpu)
    corpus_tensors = [
        torch.from_numpy(passage_embeddings[rows]).to(f"cuda:{device}", dtype=torch.float16)
        for device, rows in enumerate(shards)
    ]
    all_indices: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    for start in range(0, len(query_embeddings), 128):
        query_batch = query_embeddings[start : start + 128]
        launched = []
        for device, (rows, corpus) in enumerate(zip(shards, corpus_tensors)):
            q = torch.from_numpy(query_batch).to(f"cuda:{device}", dtype=torch.float16)
            scores = q @ corpus.T
            k = min(TOP_CHUNKS_PER_SHARD, scores.shape[1])
            values, local = torch.topk(scores, k=k, dim=1)
            launched.append((values, local, rows))
        values_np = []
        indices_np = []
        for values, local, rows in launched:
            values_np.append(values.float().cpu().numpy())
            local_np = local.cpu().numpy()
            indices_np.append(rows[local_np])
        merged_scores = np.concatenate(values_np, axis=1)
        merged_indices = np.concatenate(indices_np, axis=1)
        k = min(TOP_CHUNKS_PER_SHARD * n_gpu, merged_scores.shape[1])
        keep = np.argpartition(merged_scores, -k, axis=1)[:, -k:]
        row = np.arange(len(query_batch))[:, None]
        keep = keep[row, np.argsort(merged_scores[row, keep], axis=1)[:, ::-1]]
        all_scores.append(merged_scores[row, keep].astype(np.float32))
        all_indices.append(merged_indices[row, keep].astype(np.int32))
        print(f"dense search {min(start + 128, len(query_embeddings))}/{len(query_embeddings)}", flush=True)
    return np.concatenate(all_indices), np.concatenate(all_scores)


def gpu_local_chunks(
    queries: pd.DataFrame,
    query_embeddings: np.ndarray,
    passage_embeddings: np.ndarray,
    passage_item_rows: np.ndarray,
    items: pd.DataFrame,
) -> list[np.ndarray]:
    """Exact dense search inside each location, batching queries by location."""
    item_locations = items.item_location_id.to_numpy()
    passage_locations = item_locations[passage_item_rows]
    results: list[np.ndarray] = [np.array([], dtype=np.int32) for _ in range(len(queries))]
    for location, query_positions in queries.groupby("search_location_id").indices.items():
        passage_rows = np.flatnonzero(passage_locations == int(location))
        if len(passage_rows) == 0:
            continue
        corpus = torch.from_numpy(passage_embeddings[passage_rows]).to("cuda:0", dtype=torch.float16)
        query_positions = np.asarray(query_positions)
        for start in range(0, len(query_positions), 128):
            positions = query_positions[start : start + 128]
            q = torch.from_numpy(query_embeddings[positions]).to("cuda:0", dtype=torch.float16)
            scores = q @ corpus.T
            k = min(750, scores.shape[1])
            local = torch.topk(scores, k=k, dim=1).indices.cpu().numpy()
            for offset, position in enumerate(positions):
                results[int(position)] = passage_rows[local[offset]].astype(np.int32)
        del corpus
    return results


def item_rankings(
    queries: pd.DataFrame,
    items: pd.DataFrame,
    passage_item_rows: np.ndarray,
    chunk_indices: np.ndarray,
    local_chunk_indices: list[np.ndarray],
) -> pd.DataFrame:
    item_ids = items.item_id.astype(str).to_numpy()
    categories = items.item_category_id.to_numpy()
    locations = items.item_location_id.to_numpy()
    ratings = items.item_rating.fillna(-1).to_numpy()
    records = []
    for query_no, (_, query) in enumerate(queries.iterrows()):
        ranked_item_rows = passage_item_rows[chunk_indices[query_no]]
        seen_global: set[int] = set()
        global_ids: list[str] = []
        category = int(query.search_category)
        min_rating = requested_min_rating(query.search_infm_params_text)
        for item_row in ranked_item_rows:
            item_row = int(item_row)
            if category and categories[item_row] != category:
                continue
            if min_rating is not None and ratings[item_row] < min_rating:
                continue
            if item_row not in seen_global:
                seen_global.add(item_row)
                global_ids.append(item_ids[item_row])
            if len(global_ids) >= TOP_ITEMS:
                break
        seen_local: set[int] = set()
        local_ids: list[str] = []
        for item_row in passage_item_rows[local_chunk_indices[query_no]]:
            item_row = int(item_row)
            if category and categories[item_row] != category:
                continue
            if min_rating is not None and ratings[item_row] < min_rating:
                continue
            if item_row not in seen_local:
                seen_local.add(item_row)
                local_ids.append(item_ids[item_row])
            if len(local_ids) >= TOP_ITEMS:
                break
        records.append({
            "query_key": str(query.query_key),
            "split": query.split,
            "dense_global": " ".join(global_ids[:TOP_ITEMS]),
            "dense_local": " ".join(local_ids[:TOP_ITEMS]),
        })
    return pd.DataFrame(records)


def locate_data() -> Path:
    roots = list(Path("/kaggle/input").glob("*/benchmark_items.parquet"))
    if not roots:
        raise FileNotFoundError("benchmark_items.parquet not found under /kaggle/input")
    return roots[0].parent


def main() -> None:
    print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    data = locate_data()
    output = Path("/kaggle/working")
    item_columns = [
        "item_id", "item_title_raw", "item_description_raw", "item_infm_params_text",
        "item_category_id", "item_location_id",
        "item_rating",
    ]
    items = pd.read_parquet(data / "benchmark_items.parquet", columns=item_columns)
    benchmark = pd.read_parquet(data / "benchmark_queries.parquet")
    benchmark["query_key"] = benchmark.query_id
    benchmark["split"] = "benchmark"

    manifest_path = data / "validation_manifest.parquet"
    labels_path = data / "validation_labels.parquet"
    if not manifest_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "Upload the manifest produced by notebooks/02_validation_design.ipynb"
        )
    validation = pd.read_parquet(manifest_path).copy()
    labels = pd.read_parquet(labels_path).copy()
    validation["query_key"] = validation.eval_query_id
    validation["split"] = "validation"
    all_queries = pd.concat(
        [validation, benchmark[QUERY_COLUMNS + ["query_key", "split"]]], ignore_index=True
    )

    passages, passage_item_rows = make_passages(items)
    print(f"passages={len(passages):,}; avg/item={len(passages)/len(items):.2f}")
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = 256
    passage_embeddings = encode_multi_gpu(model, passages, batch_size=64)
    del passages
    gc.collect()
    query_embeddings = model.encode(
        [query_text(row) for _, row in all_queries.iterrows()],
        batch_size=256,
        device="cuda:0",
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    del model
    torch.cuda.empty_cache()
    chunk_indices, _ = gpu_top_chunks(query_embeddings, passage_embeddings)
    local_chunk_indices = gpu_local_chunks(
        all_queries, query_embeddings, passage_embeddings, passage_item_rows, items
    )
    rankings = item_rankings(
        all_queries, items, passage_item_rows, chunk_indices, local_chunk_indices
    )
    rankings.to_parquet(output / "dense_rankings.parquet", index=False)

    selected = labels.rename(columns={"eval_query_id": "query_key"})
    selected[["query_key", "item_id"]].drop_duplicates().to_parquet(
        output / "validation_labels.parquet", index=False
    )
    run_info = {
        "model": MODEL_NAME,
        "items": len(items),
        "passages": len(passages),
        "queries": len(all_queries),
        "chunk_words": CHUNK_WORDS,
        "overlap": CHUNK_OVERLAP,
        "max_chunks": MAX_CHUNKS,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    (output / "dense_run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(json.dumps(run_info, indent=2))


if __name__ == "__main__":
    main()
