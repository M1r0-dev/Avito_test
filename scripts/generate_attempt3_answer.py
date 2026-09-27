#!/usr/bin/env python
"""Generate answer.csv of public attempt 3 (learned fusion) by inference.

Inputs (all tracked in the repository via Git LFS, except the task data):

- `dataset/benchmark_queries.parquet`, `dataset/benchmark_items.parquet` — task data;
- `artifacts/rankings/bm25_rankings.parquet` — filter-aware BM25 channel (notebook 05);
- `artifacts/dense_kaggle/dense_rankings.parquet` — zero-shot USER-bge-m3 (Kaggle stage 04);
- `artifacts/finetuned_dense_kaggle/finetuned_dense_rankings.parquet` — LoRA USER-bge-m3
  (Kaggle stages 8A/8B);
- `models/learned_fusion_attempt3.cbm` — the learned fusion trained in notebook 17.

The script builds the candidate pool (top-100 of each channel), computes the
notebook-17 features, scores them with the saved model and writes the top-50
per query. It validates the submission contract and prints the sha256; with
`--check` it fails unless the file equals the submitted attempt 3.

Runtime: about 1-2 minutes on a laptop CPU, no GPU and no network.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from pathlib import Path

import pandas as pd
from catboost import CatBoostRanker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avito_retrieval.learned_fusion import (  # noqa: E402
    FEATURES_B, ITEM_COLUMNS, ItemSide, channel_lists, pool_of, query_features, top50,
)

ATTEMPT_3_SHA256 = "c24bf119dc311a5f333570f6fde19e55dfe40f25d578b0f347c2699388a44642"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=ROOT / "answer_attempt3.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "models/learned_fusion_attempt3.cbm")
    parser.add_argument("--check", action="store_true", help="require the submitted attempt-3 sha256")
    args = parser.parse_args()
    started = time.perf_counter()

    queries = pd.read_parquet(ROOT / "dataset/benchmark_queries.parquet")
    rankings = (
        pd.read_parquet(ROOT / "artifacts/rankings/bm25_rankings.parquet")
        .merge(pd.read_parquet(ROOT / "artifacts/dense_kaggle/dense_rankings.parquet"),
               on=["query_key", "split"], validate="one_to_one")
        .merge(pd.read_parquet(ROOT / "artifacts/finetuned_dense_kaggle/finetuned_dense_rankings.parquet"),
               on=["query_key", "split"], validate="one_to_one")
    )
    rankings = rankings[rankings.split.eq("benchmark")].set_index("query_key")
    channels = {str(key): channel_lists(row) for key, row in zip(rankings.index, rankings.itertuples())}
    assert set(channels) == set(queries.query_id.astype(str)), "rankings must cover every benchmark query"

    # Only the items that can enter a pool need offline features.
    needed = {item for lists in channels.values() for item in pool_of(lists)}
    items = pd.read_parquet(ROOT / "dataset/benchmark_items.parquet", columns=ITEM_COLUMNS)
    item_side = ItemSide.build(items[items.item_id.astype(str).isin(needed)])

    frame = pd.concat([
        query_features(query, str(query.query_id), channels[str(query.query_id)], item_side)
        for query in queries.itertuples(index=False)
    ], ignore_index=True)
    model = CatBoostRanker()
    model.load_model(str(args.model))
    predictions = top50(frame, model.predict(frame[FEATURES_B]))

    answer = pd.DataFrame({
        "query_id": queries.query_id.astype(str),
        "answer": [" ".join(predictions[str(query_id)]) for query_id in queries.query_id],
    })
    # Submission contract from the task statement.
    corpus_ids = set(items.item_id.astype(str))
    assert list(answer.columns) == ["query_id", "answer"]
    assert len(answer) == len(queries) == answer.query_id.nunique()
    assert answer.query_id.str.fullmatch(r"[A-Za-z0-9]{16}").all()
    for item_string in answer.answer:
        item_ids = item_string.split(" ")
        assert 1 <= len(item_ids) <= 50 and len(item_ids) == len(set(item_ids))
        assert all(re.fullmatch(r"[0-9a-f]{16}", item_id) for item_id in item_ids)
        assert set(item_ids) <= corpus_ids
    answer.to_csv(args.output, index=False)

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"wrote {args.output} ({len(answer)} queries) in {time.perf_counter() - started:.1f}s")
    print(f"sha256 {digest}")
    print("matches submitted attempt 3" if digest == ATTEMPT_3_SHA256 else "differs from submitted attempt 3")
    if args.check and digest != ATTEMPT_3_SHA256:
        raise SystemExit("answer.csv differs from the submitted attempt 3")


if __name__ == "__main__":
    main()
