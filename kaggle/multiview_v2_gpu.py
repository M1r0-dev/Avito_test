# %% [markdown]
# # Stage 21A — LoRA v2 multi-view retrieval (2×T4)
#
# Текущий лучший retrieval кодирует запрос вместе с человекочитаемыми
# фильтрами, а объявление — длинными passages с повторённым title. Это сильный
# общий baseline, но одно представление вынуждено одновременно сохранять
# короткий intent, фильтры и подробное описание.
#
# Здесь добавляются независимые views, не меняющие модель и разметку:
#
# - `query-only → passages`: основной intent без возможного размывания filters;
# - `query+filters → title+params`: короткое объявление без description;
# - `query-only → title+params`: самый компактный semantic match.
#
# Контроль `query+filters → passages` не пересчитывается: FP16 vectors и его
# rankings берутся из публичного stage 10B. Для честного сравнения используются
# та же LoRA v2, cosine similarity, global/location-local search и те же hard
# filters category/min-rating. В output сохраняются новые vectors, mappings и
# top-250 item rankings; решение о включении каналов принимается локально только
# по зафиксированному dev/test протоколу.

# %%
from __future__ import annotations

import gc
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, models

MODEL = "deepvk/USER-bge-m3"
MODEL_LICENSE = "apache-2.0"
QUERY_COLUMNS = [
    "search_query", "search_location_id", "search_is_delivery_search",
    "search_infm_params_text", "search_category",
]
MAX_LENGTH = 256
TOP_PER_SHARD = 1500
LOCAL_TOP = 750
TOP_ITEMS = 250
RATING_RE = re.compile(r"рейтинг[^\d]{0,20}([1-5](?:[.,]\d+)?)", re.IGNORECASE)


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).replace("ё", "е").lower().split())


def query_with_filters(row: object) -> str:
    return f"{clean(row.search_query)} {clean(row.search_infm_params_text)}".strip()


def query_only(row: object) -> str:
    return clean(row.search_query)


def short_item_text(row: object) -> str:
    """Training-compatible short view: title prefix followed by item params."""
    title = clean(row.item_title_raw)
    params = clean(row.item_infm_params_text)
    return f"{title}. {params}".strip()


def requested_min_rating(value: object) -> float | None:
    match = RATING_RE.search(str(value))
    return float(match.group(1).replace(",", ".")) if match else None


def locate_data() -> Path:
    roots = list(Path("/kaggle/input").rglob("benchmark_items.parquet"))
    if not roots:
        raise FileNotFoundError("benchmark_items.parquet not found")
    return roots[0].parent


def locate_previous() -> Path:
    files = list(Path("/kaggle/input").rglob("passage_embeddings_v2_fp16.npy"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one stage-10B vector file, found {files}")
    return files[0].parent


def locate_tuned_model() -> Path:
    candidates = []
    for metrics in Path("/kaggle/input").rglob("training_metrics.json"):
        model_dir = metrics.parent
        if (model_dir / "config.json").exists() and (model_dir / "model.safetensors").exists():
            candidates.append(model_dir)
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one merged LoRA v2 model, found {candidates}")
    return candidates[0]


def encode_multi_gpu(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    devices = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    if len(devices) != 2:
        raise RuntimeError(f"Expected two GPUs, found {devices}")
    pool = model.start_multi_process_pool(target_devices=devices)
    try:
        encoded = model.encode(
            texts, pool=pool, batch_size=batch_size, chunk_size=4096,
            normalize_embeddings=True, show_progress_bar=True,
        )
    finally:
        model.stop_multi_process_pool(pool)
    return np.asarray(encoded, dtype=np.float32)


def gpu_global(query_embeddings: np.ndarray, corpus_embeddings: np.ndarray) -> np.ndarray:
    """Exact sharded cosine top-3000, matching stage 10B."""
    shards = np.array_split(np.arange(len(corpus_embeddings)), 2)
    corpora = [
        torch.from_numpy(np.asarray(corpus_embeddings[rows])).to(
            f"cuda:{device}", dtype=torch.float16)
        for device, rows in enumerate(shards)
    ]
    all_indices = []
    for start in range(0, len(query_embeddings), 128):
        candidates, scores = [], []
        for device, (rows, corpus) in enumerate(zip(shards, corpora)):
            query = torch.from_numpy(query_embeddings[start:start + 128]).to(
                f"cuda:{device}", dtype=torch.float16)
            values, local = torch.topk(
                query @ corpus.T, k=min(TOP_PER_SHARD, corpus.shape[0]), dim=1)
            scores.append(values.float().cpu().numpy())
            candidates.append(rows[local.cpu().numpy()])
        merged_scores = np.concatenate(scores, axis=1)
        merged_indices = np.concatenate(candidates, axis=1)
        k = min(TOP_PER_SHARD * 2, merged_scores.shape[1])
        keep = np.argpartition(merged_scores, -k, axis=1)[:, -k:]
        batch_rows = np.arange(len(keep))[:, None]
        order = np.argsort(merged_scores[batch_rows, keep], axis=1)[:, ::-1]
        all_indices.append(merged_indices[batch_rows, np.take_along_axis(keep, order, axis=1)])
    del corpora
    torch.cuda.empty_cache()
    return np.concatenate(all_indices).astype(np.int32)


def gpu_local(
    queries: pd.DataFrame,
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    row_to_item: np.ndarray,
    items: pd.DataFrame,
) -> list[np.ndarray]:
    locations = items.item_location_id.to_numpy()[row_to_item]
    result = [np.array([], dtype=np.int32) for _ in range(len(queries))]
    for location, positions in queries.groupby("search_location_id").indices.items():
        corpus_rows = np.flatnonzero(locations == int(location))
        if not len(corpus_rows):
            continue
        corpus = torch.from_numpy(np.asarray(corpus_embeddings[corpus_rows])).to(
            "cuda:0", dtype=torch.float16)
        positions = np.asarray(positions)
        for start in range(0, len(positions), 128):
            batch_positions = positions[start:start + 128]
            query = torch.from_numpy(query_embeddings[batch_positions]).to(
                "cuda:0", dtype=torch.float16)
            local = torch.topk(
                query @ corpus.T, k=min(LOCAL_TOP, len(corpus_rows)), dim=1
            ).indices.cpu().numpy()
            for offset, position in enumerate(batch_positions):
                result[int(position)] = corpus_rows[local[offset]].astype(np.int32)
        del corpus
    torch.cuda.empty_cache()
    return result


def collapse_rankings(
    prefix: str,
    queries: pd.DataFrame,
    items: pd.DataFrame,
    row_to_item: np.ndarray,
    global_rows: np.ndarray,
    local_rows: list[np.ndarray],
) -> pd.DataFrame:
    item_ids = items.item_id.astype(str).to_numpy()
    categories = items.item_category_id.to_numpy()
    ratings = items.item_rating.fillna(-1).to_numpy()
    records = []
    for query_no, query in enumerate(queries.itertuples(index=False)):
        category = int(query.search_category)
        min_rating = requested_min_rating(query.search_infm_params_text)

        def collapse(rows: np.ndarray) -> str:
            answer, seen = [], set()
            for item_row in row_to_item[rows]:
                item_row = int(item_row)
                if category and categories[item_row] != category:
                    continue
                if min_rating is not None and ratings[item_row] < min_rating:
                    continue
                if item_row not in seen:
                    seen.add(item_row)
                    answer.append(item_ids[item_row])
                if len(answer) == TOP_ITEMS:
                    break
            return " ".join(answer)

        records.append({
            "query_key": str(query.query_key), "split": str(query.split),
            f"{prefix}_global": collapse(global_rows[query_no]),
            f"{prefix}_local": collapse(local_rows[query_no]),
        })
    return pd.DataFrame(records)


def retrieve_view(
    prefix: str,
    queries: pd.DataFrame,
    items: pd.DataFrame,
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    row_to_item: np.ndarray,
) -> pd.DataFrame:
    print(f"searching {prefix}", flush=True)
    global_rows = gpu_global(query_embeddings, corpus_embeddings)
    local_rows = gpu_local(queries, query_embeddings, corpus_embeddings, row_to_item, items)
    return collapse_rankings(prefix, queries, items, row_to_item, global_rows, local_rows)


# %% [markdown]
# ## Input validation
#
# Stage 10B mappings define exact vector row semantics. We assert that the raw
# task data is in the same order before any search; silent item/query reordering
# would otherwise produce syntactically valid but meaningless rankings.

# %%
def main() -> None:
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"Expected two GPUs, found {torch.cuda.device_count()}")
    data, previous, output = locate_data(), locate_previous(), Path("/kaggle/working")
    item_columns = [
        "item_id", "item_title_raw", "item_infm_params_text",
        "item_category_id", "item_location_id", "item_rating",
    ]
    items = pd.read_parquet(data / "benchmark_items.parquet", columns=item_columns)
    items["item_id"] = items.item_id.astype(str)
    manifest = pd.read_parquet(data / "validation_manifest.parquet")
    benchmark = pd.read_parquet(data / "benchmark_queries.parquet")
    benchmark["query_key"], benchmark["split"] = benchmark.query_id.astype(str), "benchmark"
    validation = manifest.copy()
    validation["query_key"], validation["split"] = validation.eval_query_id.astype(str), "validation"
    queries = pd.concat(
        [validation, benchmark[QUERY_COLUMNS + ["query_key", "split"]]], ignore_index=True)

    item_order = pd.read_parquet(previous / "item_order.parquet")
    query_order = pd.read_parquet(previous / "query_order.parquet")
    assert items.item_id.tolist() == item_order.item_id.astype(str).tolist()
    assert queries[["query_key", "split"]].astype(str).equals(query_order.astype(str))
    passage_embeddings = np.load(previous / "passage_embeddings_v2_fp16.npy", mmap_mode="r")
    passage_item_rows = np.load(previous / "passage_item_rows.npy", mmap_mode="r")
    combined_queries = np.load(previous / "query_embeddings_v2_fp16.npy").astype(np.float32)
    assert len(passage_embeddings) == len(passage_item_rows)
    assert len(combined_queries) == len(queries)

    tuned_model = locate_tuned_model()
    transformer = models.Transformer(str(tuned_model), max_seq_length=MAX_LENGTH)
    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_cls_token=True, pooling_mode_mean_tokens=False)
    model = SentenceTransformer(modules=[transformer, pooling])

    # Short corpus is the only substantial new encoding job (~34% of passages).
    short_texts = [short_item_text(row) for row in items.itertuples(index=False)]
    short_embeddings = encode_multi_gpu(model, short_texts, batch_size=64)
    query_only_embeddings = model.encode(
        [query_only(row) for row in queries.itertuples(index=False)],
        batch_size=128, device="cuda:0", normalize_embeddings=True,
        show_progress_bar=True).astype(np.float32)
    del model, short_texts
    gc.collect(); torch.cuda.empty_cache()

    np.save(output / "short_item_embeddings_v2_fp16.npy", short_embeddings.astype(np.float16))
    np.save(output / "query_only_embeddings_v2_fp16.npy", query_only_embeddings.astype(np.float16))
    items[["item_id"]].to_parquet(output / "multiview_item_order.parquet", index=False)
    queries[["query_key", "split"]].to_parquet(output / "multiview_query_order.parquet", index=False)

    identity = np.arange(len(items), dtype=np.int32)
    views = [
        retrieve_view("mv_passage_query_only", queries, items, query_only_embeddings,
                      passage_embeddings, passage_item_rows),
        retrieve_view("mv_short_combined", queries, items, combined_queries,
                      short_embeddings, identity),
        retrieve_view("mv_short_query_only", queries, items, query_only_embeddings,
                      short_embeddings, identity),
    ]
    rankings = views[0]
    for frame in views[1:]:
        rankings = rankings.merge(frame, on=["query_key", "split"], validate="one_to_one")
    rankings.to_parquet(output / "multiview_v2_rankings.parquet", index=False)

    run = {
        "stage": "multiview_retrieval_v2", "base_model": MODEL, "license": MODEL_LICENSE,
        "items": len(items), "passages": len(passage_embeddings), "queries": len(queries),
        "views": ["passage_query_only", "short_combined", "short_query_only"],
        "short_document": "normalized item_title_raw + item_infm_params_text",
        "embedding_dimension": int(short_embeddings.shape[1]), "embedding_dtype": "float16",
        "global_exact_top_rows": TOP_PER_SHARD * 2, "local_exact_top_rows": LOCAL_TOP,
        "gpus": [torch.cuda.get_device_name(i) for i in range(2)],
    }
    (output / "multiview_v2_run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run, ensure_ascii=False, indent=2), flush=True)


# %%
main()
