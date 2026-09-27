"""Text preparation and fixed-size passage construction.

The corpus contains very long descriptions and parameter strings. Retrieval is
therefore performed over passages and then aggregated back to unique item IDs.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import pandas as pd

TOKEN_RE = re.compile(r"(?u)[\w]+(?:[-'][\w]+)*")
SPACE_RE = re.compile(r"\s+")


def clean_text(value: object) -> str:
    """Normalize whitespace/case without destroying numbers or Russian text."""
    if value is None or pd.isna(value):
        return ""
    return SPACE_RE.sub(" ", str(value).replace("ё", "е").lower()).strip()


def tokenize(value: object) -> list[str]:
    """Simple deterministic tokenizer shared by BM25 and length analysis."""
    return TOKEN_RE.findall(clean_text(value))


def query_text(row: pd.Series | dict) -> str:
    """Combine free-text query with explicit human-readable filters."""
    query = clean_text(row.get("search_query", ""))
    filters = clean_text(row.get("search_infm_params_text", ""))
    return f"query: {query} {filters}".strip()


def dense_query_text(row: pd.Series | dict) -> str:
    """Query text exactly as the USER-bge-m3 Kaggle kernels (04, 10B) encoded it.

    Unlike `query_text` (BM25 and the e5 experiment) there is no "query: "
    prefix: with it the LoRA v2 query vectors reach only cosine ~0.93 to the
    saved Kaggle vectors instead of 0.999999 (notebook 28), and Recall@50 drops.
    """
    query = clean_text(row.get("search_query", ""))
    filters = clean_text(row.get("search_infm_params_text", ""))
    return f"{query} {filters}".strip()


def item_passages(
    row: pd.Series | dict,
    chunk_words: int = 140,
    overlap_words: int = 30,
    max_chunks: int = 4,
) -> Iterator[str]:
    """Yield fixed-size passages while repeating high-signal item fields.

    The title is kept as a short prefix. Structured parameters and description
    are jointly windowed: EDA shows that parameters alone can reach thousands
    of words. A cap prevents unusually long ads from dominating the index.
    """
    title = clean_text(row.get("item_title_raw", ""))
    params = tokenize(row.get("item_infm_params_text", ""))
    description = tokenize(row.get("item_description_raw", ""))
    content = [*params, "описание", *description]
    prefix = f"passage: {title}.".strip()
    if not content:
        yield prefix
        return

    step = max(1, chunk_words - overlap_words)
    for chunk_no, start in enumerate(range(0, len(content), step)):
        if chunk_no >= max_chunks:
            break
        body = " ".join(content[start : start + chunk_words])
        yield f"{prefix} {body}".strip()


def bm25_item_text(row: pd.Series | dict) -> str:
    """Build a field-weighted whole-item text for the sparse baseline.

    Repeating the short title approximates a BM25F title boost while keeping a
    single sparse index. Full structured parameters and description remain
    searchable; BM25 length normalization controls long documents.
    """
    title = clean_text(row.get("item_title_raw", ""))
    params = clean_text(row.get("item_infm_params_text", ""))
    description = clean_text(row.get("item_description_raw", ""))
    return " ".join([title, title, title, params, description]).strip()
