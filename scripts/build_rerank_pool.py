#!/usr/bin/env python
"""Build the candidate pool that the Kaggle cross-encoder stage re-scores.

Notebook 15 showed that 75.8% of the misses of attempt 2 are *selection*
misses: the positive is already in a channel list (median best rank 67) but
RRF does not lift it into the top-50. The pool is therefore the union of the
top-``DEPTH`` items of the three channels of attempt 2 (filter-aware BM25,
zero-shot dense, fine-tuned dense), built exactly as in notebook 14.

``DEPTH=100`` is the largest depth the reranker notebook may select. Smaller
depths (30/50) are nested prefixes of the same channel lists, so one GPU pass
covers every pre-registered pool size. The output contains only query texts
and item ids: raw item texts stay in the private Kaggle dataset.

Output: ``artifacts/kaggle_rerank_dataset/rerank_pool.parquet`` with one row per
(query_key, split) and columns ``query_text`` and ``candidates``
(space-separated item ids, deduplicated in channel order).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avito_retrieval.fusion import reciprocal_rank_fusion  # noqa: E402

DEPTH = 100
OUTPUT = ROOT / "artifacts/kaggle_rerank_dataset"
# Private Kaggle dataset that the reranker kernel attaches next to the raw data.
DATASET_ID = "m1r0tvorxc/avito-rerank-pool"


def parse_rank(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and value != value) or not str(value):
        return []
    return str(value).split()


def reranker_query(row: pd.Series) -> str:
    """Query text for the cross-encoder: free text plus human-readable filters.

    The same information enters BM25 and dense (`query_text`), but without the
    lower-casing and `query:` prefix: the reranker tokenizer is cased and was
    not trained with E5-style prefixes.
    """
    query = " ".join(str(row.search_query).split())
    filters = " ".join(str(row.search_infm_params_text or "").split())
    return f"{query}. {filters}" if filters else query


def main() -> None:
    rankings = (
        pd.read_parquet(ROOT / "artifacts/rankings/bm25_rankings.parquet")
        .merge(pd.read_parquet(ROOT / "artifacts/dense_kaggle/dense_rankings.parquet"),
               on=["query_key", "split"], validate="one_to_one")
        .merge(pd.read_parquet(
            ROOT / "artifacts/finetuned_dense_kaggle/finetuned_dense_rankings.parquet"),
            on=["query_key", "split"], validate="one_to_one")
    )
    manifest = pd.read_parquet(ROOT / "artifacts/validation/manifest.parquet")
    benchmark = pd.read_parquet(ROOT / "dataset/benchmark_queries.parquet")
    queries = pd.concat([
        manifest.assign(query_key=manifest.eval_query_id, split="validation"),
        benchmark.assign(query_key=benchmark.query_id, split="benchmark"),
    ], ignore_index=True)[["query_key", "split", "search_query", "search_infm_params_text"]]
    frame = rankings.merge(queries, on=["query_key", "split"], validate="one_to_one")
    assert len(frame) == len(manifest) + len(benchmark)

    rows = []
    for row in frame.itertuples(index=False):
        # Dense global/local are collapsed with the dev-selected weights of
        # notebooks 05 (1:1.15) and 08 (1:1.5), identical to attempt 2.
        zero = reciprocal_rank_fusion(
            [parse_rank(row.dense_global), parse_rank(row.dense_local)],
            weights=[1.0, 1.15], rrf_k=60, top_k=250,
        )
        fine = reciprocal_rank_fusion(
            [parse_rank(row.finetuned_global), parse_rank(row.finetuned_local)],
            weights=[1.0, 1.5], rrf_k=60, top_k=250,
        )
        pool = list(dict.fromkeys([
            *parse_rank(row.bm25)[:DEPTH], *zero[:DEPTH], *fine[:DEPTH],
        ]))
        rows.append({
            "query_key": str(row.query_key),
            "split": row.split,
            "query_text": reranker_query(row),
            "candidates": " ".join(pool),
        })
    pool_frame = pd.DataFrame(rows)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pool_frame.to_parquet(OUTPUT / "rerank_pool.parquet", index=False)
    (OUTPUT / "dataset-metadata.json").write_text(json.dumps({
        "title": "Avito Rerank Pool",
        "id": DATASET_ID,
        "licenses": [{"name": "other"}],
    }, indent=2), encoding="utf-8")
    sizes = pool_frame.candidates.str.split().str.len()
    print(f"queries={len(pool_frame):,}; pairs={int(sizes.sum()):,}; "
          f"mean pool={sizes.mean():.1f}; max pool={int(sizes.max())}")


if __name__ == "__main__":
    main()
