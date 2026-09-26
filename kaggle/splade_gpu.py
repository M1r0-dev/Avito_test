# %% [markdown]
# # Russian SPLADE candidate retrieval on 2×T4
#
# Этот notebook проверяет learned-sparse retrieval как третий, комплементарный
# канал к BM25 и dense. Основная метрика — item-level Recall@50. Мы не считаем
# английский `splade-v3` корректным baseline для русского корпуса: его BERT
# vocabulary обучен на английском. Используется `naver/neuclir22-splade-ru`,
# шестислойный SPLADE checkpoint, специально обученный для русского retrieval
# на translated MS MARCO и русских NeuCLIR/Mr.TyDi данных.
#
# До полного запуска проверяется доля `[UNK]` на русских запросах. Model card
# имеет лицензию CC BY-NC-SA 4.0; это явно фиксируется в metadata результата.

# %% [markdown]
# ## Экспериментальные факторы
#
# Все варианты получают один checkpoint и одинаковые тексты. Это позволяет
# отделить влияние инженерных решений от выбора модели:
#
# 1. Полный `q32/q64 × d64/d192` — главные эффекты и взаимодействие query и
#    document pruning без смешения двух факторов.
# 2. Победитель полного factorial выбирается только по dev Recall@50.
# 3. first chunk против максимум четырёх fixed chunks — эффект chunking именно
#    для SPLADE, а не перенос вывода из dense-эксперимента.
# 4. global против global+local — эффект географического канала.
#
# На Kaggle сохраняются широкие top-250 списки. Выбор pruning, RRF-весов и
# `rrf_k` выполняется позже только на dev-половине. Untouched test используется
# один раз для paired bootstrap CI и paired sign-randomization test.
#
# Конкретные значения не подбираются по test. `140/30/4` перенесены из EDA и
# предыдущей подтверждённой dense-абляции: длинные описания требуют нескольких
# passages, а четыре chunks значимо лучше первого. Мы всё равно повторно
# проверяем этот вывод для SPLADE. `q32/q64` и `d64/d192` образуют маленькую
# факторную абляцию «качество против размера sparse-индекса»: запросы короче
# документов, поэтому им заранее отведён меньший бюджет термов. Сначала
# сохраняется наиболее широкий вариант, более узкие получаются только
# детерминированным top-weight pruning — без повторного inference и без
# изменения checkpoint.

# %%
from __future__ import annotations

import gc
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sentence_transformers import SparseEncoder
from transformers import AutoTokenizer

MODEL = "naver/neuclir22-splade-ru"
MODEL_LICENSE = "cc-by-nc-sa-4.0"
QUERY_COLUMNS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]
CHUNK_WORDS = 140
CHUNK_OVERLAP = 30
MAX_CHUNKS = 4
MAX_LENGTH = 256
DOC_MAX_DIMS = 192
QUERY_MAX_DIMS = 64
TOP_CHUNKS = 4000
TOP_ITEMS = 250
TOKEN_RE = re.compile(r"(?u)[\w]+(?:[-'][\w]+)*")
RATING_RE = re.compile(r"рейтинг[^\d]{0,20}([1-5](?:[.,]\d+)?)", re.IGNORECASE)


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).replace("ё", "е").lower().split())


def tokenize(value: object) -> list[str]:
    return TOKEN_RE.findall(clean(value))


def make_passages(items: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
    passages: list[str] = []
    item_rows: list[int] = []
    chunk_numbers: list[int] = []
    step = CHUNK_WORDS - CHUNK_OVERLAP
    for item_row, row in enumerate(items.itertuples(index=False)):
        title = clean(row.item_title_raw)
        content = tokenize(row.item_infm_params_text) + ["описание"] + tokenize(row.item_description_raw)
        if not content:
            content = [""]
        for chunk_no, start in enumerate(range(0, len(content), step)):
            if chunk_no >= MAX_CHUNKS:
                break
            passages.append(f"{title}. {' '.join(content[start:start + CHUNK_WORDS])}".strip())
            item_rows.append(item_row)
            chunk_numbers.append(chunk_no)
    return passages, np.asarray(item_rows, dtype=np.int32), np.asarray(chunk_numbers, dtype=np.int8)


def query_text(row: pd.Series) -> str:
    return f"{clean(row.search_query)} {clean(row.search_infm_params_text)}".strip()


def requested_min_rating(value: object) -> float | None:
    match = RATING_RE.search(str(value))
    return float(match.group(1).replace(",", ".")) if match else None


def torch_sparse_to_csr(tensor: torch.Tensor) -> sp.csr_matrix:
    tensor = tensor.coalesce().cpu()
    rows, cols = tensor.indices().numpy()
    values = tensor.values().float().numpy()
    return sp.coo_matrix((values, (rows, cols)), shape=tuple(tensor.shape), dtype=np.float32).tocsr()


def prune_csr(matrix: sp.csr_matrix, max_dims: int) -> sp.csr_matrix:
    """Keep the largest positive SPLADE dimensions independently per row."""
    matrix = matrix.tocsr()
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row : row + 2]
        cols = matrix.indices[start:end]
        values = matrix.data[start:end]
        if len(values) > max_dims:
            keep = np.argpartition(values, -max_dims)[-max_dims:]
            cols, values = cols[keep], values[keep]
        row_parts.append(np.full(len(values), row, dtype=np.int32))
        col_parts.append(cols.astype(np.int32, copy=False))
        value_parts.append(values.astype(np.float32, copy=False))
    rows = np.concatenate(row_parts)
    cols = np.concatenate(col_parts)
    values = np.concatenate(value_parts)
    return sp.coo_matrix((values, (rows, cols)), shape=matrix.shape).tocsr()


def ranked_items(
    passage_indices: np.ndarray,
    scores: np.ndarray,
    query: pd.Series,
    items: pd.DataFrame,
    passage_item_rows: np.ndarray,
    *,
    local: bool,
) -> list[str]:
    if len(scores) == 0:
        return []
    item_rows = passage_item_rows[passage_indices]
    allowed = np.ones(len(scores), dtype=bool)
    category = int(query.search_category)
    if category:
        allowed &= items.item_category_id.to_numpy()[item_rows] == category
    min_rating = requested_min_rating(query.search_infm_params_text)
    if min_rating is not None:
        allowed &= items.item_rating.fillna(-1).to_numpy()[item_rows] >= min_rating
    if local and not bool(query.search_is_delivery_search):
        allowed &= items.item_location_id.to_numpy()[item_rows] == int(query.search_location_id)
    elif local:
        return []

    passage_indices = passage_indices[allowed]
    scores = scores[allowed]
    if len(scores) == 0:
        return []
    k = min(TOP_CHUNKS, len(scores))
    keep = np.argpartition(scores, -k)[-k:]
    # Passage row is the deterministic tie-breaker.
    keep = keep[np.lexsort((passage_indices[keep], -scores[keep]))]
    item_ids = items.item_id.astype(str).to_numpy()
    result: list[str] = []
    seen: set[int] = set()
    for passage_index in passage_indices[keep]:
        item_row = int(passage_item_rows[int(passage_index)])
        if item_row not in seen:
            seen.add(item_row)
            result.append(item_ids[item_row])
            if len(result) == TOP_ITEMS:
                break
    return result


def retrieve_variant(
    queries: pd.DataFrame,
    items: pd.DataFrame,
    documents: sp.csr_matrix,
    query_matrix: sp.csr_matrix,
    passage_item_rows: np.ndarray,
    prefix: str,
    batch_size: int = 16,
) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    document_t = documents.T.tocsc()
    for start in range(0, len(queries), batch_size):
        batch_scores = (query_matrix[start : start + batch_size] @ document_t).tocsr()
        for offset in range(batch_scores.shape[0]):
            query = queries.iloc[start + offset]
            left, right = batch_scores.indptr[offset : offset + 2]
            passage_indices = batch_scores.indices[left:right]
            scores = batch_scores.data[left:right]
            global_ids = ranked_items(
                passage_indices, scores, query, items, passage_item_rows, local=False
            )
            local_ids = ranked_items(
                passage_indices, scores, query, items, passage_item_rows, local=True
            )
            records.append(
                {
                    "query_key": str(query.query_key),
                    "split": str(query.split),
                    f"{prefix}_global": " ".join(global_ids),
                    f"{prefix}_local": " ".join(local_ids),
                }
            )
        print(f"{prefix} search {min(start + batch_size, len(queries))}/{len(queries)}", flush=True)
    del document_t
    gc.collect()
    return pd.DataFrame(records)


def locate_data() -> Path:
    roots = list(Path("/kaggle/input").rglob("benchmark_items.parquet"))
    if not roots:
        raise FileNotFoundError("benchmark_items.parquet was not found under /kaggle/input")
    print("Dataset mounted at", roots[0].parent, flush=True)
    return roots[0].parent


# %% [markdown]
# ## Полный запуск и экспорт широких rankings
#
# SPLADE scores не нормализуются: это положительные learned lexical weights,
# для которых корректен dot product. Сначала кодируем `d192/q64`, затем
# детерминированно получаем более агрессивные варианты pruning без повторного
# прогона модели. Exact sparse matrix multiplication исключает ANN-погрешность.

# %%
def main() -> None:
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if len(devices) != 2:
        raise RuntimeError(f"Expected exactly two T4 GPUs, got {devices}")
    print("GPUs:", [torch.cuda.get_device_name(i) for i in range(2)], flush=True)

    data = locate_data()
    output = Path("/kaggle/working")
    item_columns = [
        "item_id",
        "item_title_raw",
        "item_description_raw",
        "item_infm_params_text",
        "item_category_id",
        "item_location_id",
        "item_rating",
    ]
    items = pd.read_parquet(data / "benchmark_items.parquet", columns=item_columns)
    benchmark = pd.read_parquet(data / "benchmark_queries.parquet")
    benchmark["query_key"] = benchmark.query_id
    benchmark["split"] = "benchmark"
    validation = pd.read_parquet(data / "validation_manifest.parquet").copy()
    labels = pd.read_parquet(data / "validation_labels.parquet").copy()
    validation["query_key"] = validation.eval_query_id
    validation["split"] = "validation"
    queries = pd.concat(
        [validation, benchmark[QUERY_COLUMNS + ["query_key", "split"]]], ignore_index=True
    )

    passages, passage_item_rows, chunk_numbers = make_passages(items)
    first_mask = chunk_numbers == 0
    print(f"passages={len(passages):,}; avg/item={len(passages)/len(items):.2f}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    language_sample = queries.search_query.astype(str).sample(
        min(1000, len(queries)), random_state=42
    )
    token_ids = [tokenizer(text, add_special_tokens=False)["input_ids"] for text in language_sample]
    token_count = sum(map(len, token_ids))
    unk_count = sum(sum(token == tokenizer.unk_token_id for token in row) for row in token_ids)
    unk_rate = unk_count / max(token_count, 1)
    print(f"Russian tokenizer UNK rate={unk_rate:.6f} ({unk_count}/{token_count})", flush=True)

    model = SparseEncoder(
        MODEL,
        model_kwargs={"torch_dtype": torch.float16, "attn_implementation": "eager"},
        max_active_dims=DOC_MAX_DIMS,
    )
    model.max_seq_length = MAX_LENGTH
    pool = model.start_multi_process_pool(target_devices=devices)
    try:
        document_tensor = model.encode_document(
            passages,
            pool=pool,
            batch_size=32,
            chunk_size=2048,
            max_active_dims=DOC_MAX_DIMS,
            show_progress_bar=True,
            convert_to_tensor=True,
            convert_to_sparse_tensor=True,
        )
        query_tensor = model.encode_query(
            [query_text(row) for _, row in queries.iterrows()],
            pool=pool,
            batch_size=128,
            chunk_size=512,
            max_active_dims=QUERY_MAX_DIMS,
            show_progress_bar=True,
            convert_to_tensor=True,
            convert_to_sparse_tensor=True,
        )
    finally:
        model.stop_multi_process_pool(pool)
    del model, tokenizer, passages
    gc.collect()

    documents_192 = torch_sparse_to_csr(document_tensor)
    queries_64 = torch_sparse_to_csr(query_tensor)
    del document_tensor, query_tensor
    documents_64 = prune_csr(documents_192, 64)
    queries_32 = prune_csr(queries_64, 32)
    print(
        "nnz/row:",
        {
            "d192": documents_192.nnz / documents_192.shape[0],
            "d64": documents_64.nnz / documents_64.shape[0],
            "q64": queries_64.nnz / queries_64.shape[0],
            "q32": queries_32.nnz / queries_32.shape[0],
        },
        flush=True,
    )

    variants = [
        ("splade_q32_d64", documents_64, queries_32, passage_item_rows),
        ("splade_q64_d64", documents_64, queries_64, passage_item_rows),
        ("splade_q32_d192", documents_192, queries_32, passage_item_rows),
        ("splade_q64_d192", documents_192, queries_64, passage_item_rows),
        (
            "splade_first_q32_d192",
            documents_192[first_mask],
            queries_32,
            passage_item_rows[first_mask],
        ),
    ]
    rankings: pd.DataFrame | None = None
    for prefix, document_matrix, query_matrix, item_map in variants:
        current = retrieve_variant(
            queries, items, document_matrix, query_matrix, item_map, prefix=prefix
        )
        rankings = (
            current
            if rankings is None
            else rankings.merge(current, on=["query_key", "split"], validate="one_to_one")
        )
    assert rankings is not None and len(rankings) == len(queries)
    rankings.to_parquet(output / "splade_rankings.parquet", index=False)
    labels[["eval_query_id", "item_id"]].drop_duplicates().to_parquet(
        output / "validation_labels.parquet", index=False
    )
    metadata = {
        "model": MODEL,
        "license": MODEL_LICENSE,
        "items": len(items),
        "passages": len(passage_item_rows),
        "queries": len(queries),
        "chunk_words": CHUNK_WORDS,
        "overlap": CHUNK_OVERLAP,
        "max_chunks": MAX_CHUNKS,
        "max_length": MAX_LENGTH,
        "document_max_dims": DOC_MAX_DIMS,
        "query_max_dims": QUERY_MAX_DIMS,
        "tokenizer_unk_rate": unk_rate,
        "gpus": [torch.cuda.get_device_name(i) for i in range(2)],
    }
    (output / "splade_run.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

# %% [markdown]
# ## Что будет проверяться локально
#
# Этот notebook намеренно не выбирает победителя по validation целиком.
# `notebooks/06_splade_experiments.ipynb` использует тот же заранее заданный
# dev/test split, что dense/RRF:
#
# - выбирает pruning и RRF grid только по dev Recall@50;
# - на test оценивает pruning, chunking, local-channel и добавление SPLADE;
# - для каждого решения сообщает mean paired delta, 95% bootstrap CI и
#   one-sided paired randomization p-value;
# - новый `answer.csv` принимается только при подтверждённом улучшении.
