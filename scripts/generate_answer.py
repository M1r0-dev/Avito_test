#!/usr/bin/env python
"""Generate answer.csv of the final solution (public attempt 7) by inference.

Pipeline: BM25 + zero-shot USER-bge-m3 + LoRA v2 USER-bge-m3 rankings ->
union of top-100 per channel -> RRF top-200 pool -> notebook-19 features ->
three PU-bagged CatBoost models -> mean reciprocal rank -> top-50.

`--objective recall50` (default) is attempt 7: the bags are trained with the
Recall@50 lambda objective (`avito_retrieval.recall_lambda`, public 0.840831).
`--objective yetirank` reproduces attempt 6 (YetiRankPairwise, public 0.836972).

Inputs (tracked via Git LFS, except the raw task data in `dataset/`):

- `artifacts/rankings/bm25_rankings.parquet` — filter-aware BM25 (notebook 05);
- `artifacts/dense_kaggle/dense_rankings.parquet` — zero-shot USER-bge-m3 (Kaggle 04);
- `artifacts/finetuned_v2_kaggle/finetuned_v2_dense_rankings.parquet` — LoRA v2 (Kaggle 10B);
- `models/recall50_seed{41,42,43}_round{1,2}.cbm` (attempt 7) or
  `models/pu_selector_seed{41,42,43}.cbm` (attempt 6) — `scripts/train_pu_selector.py`.

With `--check` the script fails unless the file equals the submitted attempt.
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
from avito_retrieval.recall_lambda import TREES, ROUND_TREES, SummedRanker  # noqa: E402

SUBMITTED = {  # objective -> (public attempt, sha256 of the submitted file)
    "recall50": (7, "994ef6f382202168e2a3a0bb6cc7673823a503af47b6d5b88e231e99d2ad7f0c"),
    "yetirank": (6, "27705d5bc6a66608b3737d54f299bbdeb70a548193bb066b118899b4e3bfb1e3"),
}


def load_models(models_dir: Path, objective: str) -> list:
    models = []
    for seed in PU_SEEDS:
        if objective == "yetirank":
            model = CatBoostRanker()
            model.load_model(str(models_dir / f"pu_selector_seed{seed}.cbm"))
            models.append(model)
            continue
        rounds = []
        for number in range(1, TREES // ROUND_TREES + 1):
            model = CatBoostRanker()
            model.load_model(str(models_dir / f"recall50_seed{seed}_round{number}.cbm"))
            rounds.append(model)
        models.append(SummedRanker(rounds))
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=ROOT / "answer.csv")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--objective", choices=tuple(SUBMITTED), default="recall50")
    parser.add_argument("--check", action="store_true", help="require the sha256 of the submitted attempt")
    args = parser.parse_args()
    started = time.perf_counter()

    queries = pd.read_parquet(ROOT / "dataset/benchmark_queries.parquet")
    items = pd.read_parquet(ROOT / "dataset/benchmark_items.parquet", columns=ITEM_COLUMNS)
    channels = load_channels(ROOT, "benchmark")
    assert set(channels) == set(queries.query_id.astype(str)), "rankings must cover every benchmark query"
    frame = build_features(queries, "query_id", channels, items)

    predictions = ensemble_top50(frame, load_models(args.models_dir, args.objective))

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
    attempt, expected = SUBMITTED[args.objective]
    print(f"wrote {args.output} ({len(answer)} queries) in {time.perf_counter() - started:.1f}s")
    print(f"sha256 {digest}")
    print(f"matches submitted attempt {attempt}" if digest == expected else f"differs from submitted attempt {attempt}")
    if args.check and digest != expected:
        raise SystemExit(f"answer.csv differs from the submitted attempt {attempt}")


if __name__ == "__main__":
    main()
