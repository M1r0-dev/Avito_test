#!/usr/bin/env python
"""Evaluate BM25 on historical positives that remain in benchmark corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avito_retrieval.bm25 import SparseBM25  # noqa: E402
from avito_retrieval.channels import filtered_ranking_from_scores  # noqa: E402
from avito_retrieval.metrics import hit_rate_at_k, recall_at_k  # noqa: E402
from avito_retrieval.language import dominant_script  # noqa: E402
from avito_retrieval.text import query_text  # noqa: E402

QUERY_COLUMNS = [
    "search_query", "search_location_id", "search_is_delivery_search",
    "search_infm_params_text", "search_category",
]


def stable_query_id(row: pd.Series) -> str:
    payload = "\x1f".join(str(row[c]) for c in QUERY_COLUMNS)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=ROOT / "artifacts/bm25")
    parser.add_argument("--max-queries", type=int, default=2452)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/bm25_metrics.json")
    args = parser.parse_args()

    items = pd.read_parquet(args.index / "items.parquet")
    language_path = args.index / "item_language.parquet"
    if language_path.exists():
        item_languages = pd.read_parquet(language_path).set_index("item_id").lang_script
        items["lang_script"] = items.item_id.map(item_languages).fillna("other")
    index = SparseBM25.load(args.index)
    manifest_path = ROOT / "artifacts/validation/manifest.parquet"
    labels_path = ROOT / "artifacts/validation/labels.parquet"
    if not manifest_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "Run notebooks/02_validation_design.ipynb before evaluation"
        )
    queries = pd.read_parquet(manifest_path).head(args.max_queries)
    labeled = pd.read_parquet(labels_path)
    labeled = labeled[labeled.eval_query_id.isin(set(queries.eval_query_id))]
    relevant = labeled.groupby("eval_query_id").item_id.agg(set).to_dict()

    predictions: dict[str, list[str]] = {}
    predictions_cyrillic_only: dict[str, list[str]] = {}
    batch_size = 64
    queries = queries.reset_index(drop=True)
    for start in range(0, len(queries), batch_size):
        batch = queries.iloc[start : start + batch_size]
        score_matrix = index.score_many([query_text(row) for _, row in batch.iterrows()])
        for column, (_, query) in enumerate(batch.iterrows()):
            predictions[query.eval_query_id] = filtered_ranking_from_scores(
                query, items, score_matrix[:, column], retrieve_k=250, output_k=50
            )
            if "lang_script" in items and dominant_script(query.search_query) == "cyrillic":
                language_mask = items.lang_script.eq("cyrillic").to_numpy()
            else:
                language_mask = None
            predictions_cyrillic_only[query.eval_query_id] = filtered_ranking_from_scores(
                query, items, score_matrix[:, column], retrieve_k=250, output_k=50,
                extra_mask=language_mask,
            )
        print(f"Retrieved {min(start + batch_size, len(queries))}/{len(queries)}", flush=True)

    metrics = {
        "queries": len(queries),
        "corpus_items": len(items),
        "recall@50": recall_at_k(predictions, relevant, 50),
        "hit_rate@50": hit_rate_at_k(predictions, relevant, 50),
        "cyrillic_only_recall@50": recall_at_k(predictions_cyrillic_only, relevant, 50),
        "cyrillic_only_hit_rate@50": hit_rate_at_k(predictions_cyrillic_only, relevant, 50),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
