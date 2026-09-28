#!/usr/bin/env python
"""Train the PU-bagged selector of the final solution (public attempt 7, or attempt 6).

Inputs are tracked in the repository (Git LFS), except the raw task data:

- `artifacts/validation/{manifest,labels}.parquet` — the fixed holdout (notebook 02);
- `artifacts/rankings/bm25_rankings.parquet` — filter-aware BM25 (notebook 05);
- `artifacts/dense_kaggle/dense_rankings.parquet` — zero-shot USER-bge-m3 (Kaggle 04);
- `artifacts/finetuned_v2_kaggle/finetuned_v2_dense_rankings.parquet` — LoRA v2 (Kaggle 10B);
- `dataset/benchmark_items.parquet` — item texts and priors.

The selector is trained on the dev half of the holdout only, exactly as in
notebook 24 (the test half was used once for the statistical gate). The script
rebuilds the notebook-19 features with `avito_retrieval.pu_selector`, and with
`--verify-export` compares them to the notebook-25A export of notebook 19
before training, so the module port is checked rather than assumed.
Two objectives inside the same three PU bags:

- `--objective recall50` (default, attempt 7): LambdaMART with |ΔRecall@50|
  pair weights over PairLogit (`avito_retrieval.recall_lambda`), two rounds of
  50 trees per bag -> `models/recall50_seed{41,42,43}_round{1,2}.cbm`;
- `--objective yetirank` (attempt 6): YetiRankPairwise, 300 trees per bag ->
  `models/pu_selector_seed{41,42,43}.cbm`.

About 1-2 minutes on a laptop CPU, mostly feature building.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avito_retrieval.learned_fusion import ITEM_COLUMNS  # noqa: E402
from avito_retrieval.pu_selector import (  # noqa: E402
    FEATURES, PU_SEEDS, build_features, load_channels, train_pu_models,
)
from avito_retrieval.recall_lambda import train_models as train_recall50_models  # noqa: E402

RANDOM_SEED = 42  # holdout dev/test split of notebooks 17-24


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--objective", choices=("recall50", "yetirank"), default="recall50")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--verify-export", type=Path, default=None,
                        help="c2_validation.parquet of notebook 25A to compare features with")
    args = parser.parse_args()
    started = time.perf_counter()

    manifest = pd.read_parquet(ROOT / "artifacts/validation/manifest.parquet")
    labels = pd.read_parquet(ROOT / "artifacts/validation/labels.parquet")
    folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_SEED)
    dev_index = next(iter(folds.split(manifest, manifest.primary_stratum)))[1]
    dev = manifest.iloc[dev_index].reset_index(drop=True)

    items = pd.read_parquet(ROOT / "dataset/benchmark_items.parquet", columns=ITEM_COLUMNS)
    frame = build_features(dev, "eval_query_id", load_channels(ROOT, "validation"), items)
    positives = set(zip(labels.eval_query_id.astype(str), labels.item_id.astype(str)))
    frame["label"] = [int((q, item) in positives) for q, item in zip(frame.query_key, frame.item_id)]
    print(f"dev features {frame.shape}, positives {int(frame.label.sum())}", flush=True)

    if args.verify_export is not None:
        export = pd.read_parquet(args.verify_export)
        export = export[export.query_key.isin(set(dev.eval_query_id))].reset_index(drop=True)
        columns = ["query_key", "item_id", "label", *FEATURES]
        pd.testing.assert_frame_equal(frame[columns], export[columns], check_dtype=False)
        print("features identical to the notebook-19 export", flush=True)

    args.models_dir.mkdir(parents=True, exist_ok=True)
    if args.objective == "yetirank":
        for seed, model in zip(PU_SEEDS, train_pu_models(frame)):
            path = args.models_dir / f"pu_selector_seed{seed}.cbm"
            model.save_model(str(path))
            print("saved", path, flush=True)
    else:
        # Denominator of macro Recall@50: all relevant items of the query,
        # including those the pool missed (notebook 25A export does the same).
        n_relevant = labels.groupby("eval_query_id").size().to_dict()
        for seed, rounds in zip(PU_SEEDS, train_recall50_models(frame, n_relevant)):
            for number, model in enumerate(rounds, 1):
                path = args.models_dir / f"recall50_seed{seed}_round{number}.cbm"
                model.save_model(str(path))
                print("saved", path, flush=True)
    print(f"done in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()
