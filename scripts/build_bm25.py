#!/usr/bin/env python
"""Build the benchmark-corpus BM25 index and compact item metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avito_retrieval.bm25 import SparseBM25  # noqa: E402
from avito_retrieval.text import bm25_item_text  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, default=ROOT / "dataset/benchmark_items.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/bm25")
    args = parser.parse_args()

    columns = [
        "item_id", "item_title_raw", "item_description_raw", "item_infm_params_text",
        "item_category_id", "item_location_id", "item_rating",
    ]
    items = pd.read_parquet(args.items, columns=columns)
    if not items.item_id.is_unique:
        raise ValueError("Corpus item_id must be unique")
    documents = [bm25_item_text(row) for _, row in items.iterrows()]
    index = SparseBM25().fit(documents)
    index.save(args.output)
    items[["item_id", "item_category_id", "item_location_id", "item_rating"]].to_parquet(
        args.output / "items.parquet", index=False
    )
    print(f"Indexed {len(items):,} items; vocabulary={index.matrix.shape[1]:,}")


if __name__ == "__main__":
    main()

