#!/usr/bin/env python
"""Generate answer.csv of the final solution (public attempt 6) by inference.

Pipeline: BM25 + zero-shot USER-bge-m3 + LoRA v2 USER-bge-m3 rankings ->
union of top-100 per channel -> RRF top-200 pool -> notebook-19 features ->
three PU-bagged CatBoost models -> mean reciprocal rank -> top-50.

Inputs (tracked via Git LFS, except the raw task data in `dataset/`):

- `artifacts/rankings/bm25_rankings.parquet` — filter-aware BM25 (notebook 05);
- `artifacts/dense_kaggle/dense_rankings.parquet` — zero-shot USER-bge-m3 (Kaggle 04);
- `artifacts/finetuned_v2_kaggle/finetuned_v2_dense_rankings.parquet` — LoRA v2 (Kaggle 10B);
- `models/pu_selector_seed{41,42,43}.cbm` — `scripts/train_pu_selector.py` (notebook 24).

With `--check` the script fails unless the file equals the submitted attempt 6.
Runtime: about 1 minute on a laptop CPU, no GPU and no network.
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

from avito_retrieval.learned_fusion import ITEM_COLUMNS  # noqa: E402
from avito_retrieval.pu_selector import PU_SEEDS, build_features, ensemble_top50, load_channels  # noqa: E402

ATTEMPT_6_SHA256 = "27705d5bc6a66608b3737d54f299bbdeb70a548193bb066b118899b4e3bfb1e3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=ROOT / "answer.csv")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--check", action="store_true", help="require the submitted attempt-6 sha256")
    args = parser.parse_args()
    started = time.perf_counter()

    queries = pd.read_parquet(ROOT / "dataset/benchmark_queries.parquet")
    items = pd.read_parquet(ROOT / "dataset/benchmark_items.parquet", columns=ITEM_COLUMNS)
    channels = load_channels(ROOT, "benchmark")
    assert set(channels) == set(queries.query_id.astype(str)), "rankings must cover every benchmark query"
    frame = build_features(queries, "query_id", channels, items)

    models = []
    for seed in PU_SEEDS:
        model = CatBoostRanker()
        model.load_model(str(args.models_dir / f"pu_selector_seed{seed}.cbm"))
        models.append(model)
    predictions = ensemble_top50(frame, models)

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
        assert len(item_ids) == 50 == len(set(item_ids))
        assert all(re.fullmatch(r"[0-9a-f]{16}", item_id) for item_id in item_ids)
        assert set(item_ids) <= corpus_ids
    answer.to_csv(args.output, index=False)

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"wrote {args.output} ({len(answer)} queries) in {time.perf_counter() - started:.1f}s")
    print(f"sha256 {digest}")
    print("matches submitted attempt 6" if digest == ATTEMPT_6_SHA256 else "differs from submitted attempt 6")
    if args.check and digest != ATTEMPT_6_SHA256:
        raise SystemExit("answer.csv differs from the submitted attempt 6")


if __name__ == "__main__":
    main()
