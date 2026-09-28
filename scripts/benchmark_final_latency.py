#!/usr/bin/env python
"""Warm batch=1 CPU latency of the final candidate generator (public attempt 7).

Online path per query, as in `scripts/generate_answer.py` but starting from
the raw query instead of saved rankings:

    normalize -> BM25 (filtered + plain) ->
    zero-shot USER-bge-m3: encode -> exact global/local passage search ->
    LoRA v2 USER-bge-m3:   encode -> exact global/local passage search ->
    RRF top-200 pool -> selector features -> 3 PU CatBoost bags -> top-50

`--objective recall50` (default) loads the attempt-7 selector (two 50-tree
rounds per bag); `--objective yetirank` the attempt-6 one (300 trees per bag).

`--modes sequential` runs the stages one after another; `parallel` runs the
three independent retrieval branches (BM25, zero-shot, LoRA v2) in threads:
PyTorch/ONNX Runtime, NumPy/BLAS and CatBoost release the GIL. When several
modes are measured the script asserts identical top-50 for every query.

`--encoder torch` is SentenceTransformer; `--encoder onnx` is the fp32 ONNX
Runtime export (`scripts/export_onnx_encoder.py`), whose rankings are
identical to PyTorch (notebook 28).

Fidelity notes (same approach as `benchmark_pipeline_latency.py`):
- LoRA v2 searches the real saved passage vectors (Kaggle stage 10B);
- zero-shot passage vectors were not saved, so its index is a random unit
  matrix of the real shape — exact search costs the same for any values;
- with `--encoder onnx` the LoRA v2 branch runs the real merged LoRA v2 export;
  with `--encoder torch` both branches use the base USER-bge-m3 weights
  (identical architecture; merged LoRA adds no inference cost).
Recall of the CPU dense path itself is checked in notebook 27.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

# Intel OpenMP (MKL and PyTorch here) spins for 200 ms after each parallel
# region by default; with concurrent branches the spinning threads starve the
# other branch and every stage gets a heavy tail. Must be set before numpy.
os.environ.setdefault("KMP_BLOCKTIME", "0")
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from avito_retrieval.bm25 import SparseBM25  # noqa: E402
from avito_retrieval.channels import filtered_ranking_from_scores, top_rows  # noqa: E402
from avito_retrieval.dense_search import ExactDenseIndex  # noqa: E402
from avito_retrieval.filters import requested_min_rating  # noqa: E402
from avito_retrieval.learned_fusion import ITEM_COLUMNS, ItemSide  # noqa: E402
from avito_retrieval.onnx_encoder import MAX_LENGTH, OnnxQueryEncoder  # noqa: E402
from avito_retrieval.pu_selector import (  # noqa: E402
    V2_LOCAL_WEIGHT, ZERO_LOCAL_WEIGHT, channel_lists, ensemble_top50, pool_items, query_features,
)
from avito_retrieval.text import dense_query_text, query_text  # noqa: E402
from benchmark_latency import hardware, summarize  # noqa: E402
from generate_answer import load_models  # noqa: E402

MODEL = "deepvk/USER-bge-m3"
MODEL_REVISION = "0cc6cfe48e260fb0474c753087a69369e88709ae"
V2_DIR = ROOT / "artifacts/finetuned_v2_kaggle"
# `scripts/export_onnx_encoder.py` (base) and `--model <merged LoRA v2> --output artifacts/onnx/lora_v2`.
ONNX_DIRS = {"zero": ROOT / "artifacts/onnx/user_bge_m3", "v2": ROOT / "artifacts/onnx/lora_v2"}
STAGES = ("bm25", "encode_zero", "search_zero", "encode_v2", "search_v2", "features", "selector")


class Pipeline:
    def __init__(self, encoder: str, threads: int, seed: int, blas_threads: int, catboost_threads: int,
                 objective: str) -> None:
        torch.set_num_threads(threads)
        # Thread budget per concurrent branch: encoder `threads`, exact search
        # `blas_threads` (MKL matmul), selector `catboost_threads`. None of these
        # changes any score; `--compare` of modes checks the top-50 stays identical.
        threadpool_limits(limits=blas_threads, user_api="blas")
        self.catboost_threads = catboost_threads
        self.index = SparseBM25.load(ROOT / "artifacts/bm25")
        self.items = pd.read_parquet(ROOT / "artifacts/bm25/items.parquet")
        item_ids = self.items.item_id.astype(str).to_numpy()
        passage_rows = np.load(V2_DIR / "passage_item_rows.npy")
        args = (item_ids, self.items.item_location_id.to_numpy(), self.items.item_category_id.to_numpy(),
                self.items.item_rating.fillna(-1).to_numpy())
        v2 = np.load(V2_DIR / "passage_embeddings_v2_fp16.npy")
        rng = np.random.default_rng(seed)
        zero = rng.standard_normal(v2.shape, dtype=np.float32)
        zero /= np.linalg.norm(zero, axis=1, keepdims=True)
        self.dense = {"zero": ExactDenseIndex(zero, passage_rows, *args, ZERO_LOCAL_WEIGHT),
                      "v2": ExactDenseIndex(v2, passage_rows, *args, V2_LOCAL_WEIGHT)}
        del zero, v2
        self.encoders = {}
        for name in self.dense:
            if encoder == "onnx":
                self.encoders[name] = OnnxQueryEncoder(ONNX_DIRS[name], threads, allow_spinning=False).encode
            else:
                model = SentenceTransformer(MODEL, revision=MODEL_REVISION, device="cpu")
                model.max_seq_length = MAX_LENGTH  # as in the Kaggle kernels
                self.encoders[name] = (lambda text, model=model: model.encode(
                    text, normalize_embeddings=True, convert_to_numpy=True))
        raw_items = pd.read_parquet(ROOT / "dataset/benchmark_items.parquet", columns=ITEM_COLUMNS)
        self.item_side = ItemSide.build(raw_items)  # offline item cache, all 189k items
        self.models = load_models(ROOT / "models", objective)
        self.all_items = np.ones(len(self.items), dtype=bool)

    def bm25(self, query: pd.Series, text: str) -> tuple[list[str], list[str], float]:
        started = time.perf_counter_ns()
        scores = self.index.score(text)
        filtered = filtered_ranking_from_scores(query, self.items, scores, retrieve_k=300, output_k=250)
        plain = self.items.item_id.iloc[top_rows(scores, self.all_items, 250)].astype(str).tolist()
        return filtered, plain, (time.perf_counter_ns() - started) / 1e6

    def dense_branch(self, name: str, query: pd.Series, text: str):
        started = time.perf_counter_ns()
        with torch.inference_mode():
            vector = self.encoders[name](text)
        encoded = time.perf_counter_ns()
        lists = self.dense[name].search(vector, int(query.search_category),
                                        requested_min_rating(query.search_infm_params_text),
                                        int(query.search_location_id))
        done = time.perf_counter_ns()
        return lists, (encoded - started) / 1e6, (done - encoded) / 1e6

    def run(self, query: pd.Series, pool: ThreadPoolExecutor | None) -> tuple[list[str], dict[str, float]]:
        timings = {}
        # BM25 rankings were built from `query_text`; the USER-bge-m3 kernels
        # encoded `dense_query_text` (no "query: " prefix, notebook 28).
        text, dense_text = query_text(query), dense_query_text(query)
        if pool is None:
            bm25, plain, timings["bm25"] = self.bm25(query, text)
            zero, timings["encode_zero"], timings["search_zero"] = self.dense_branch("zero", query, dense_text)
            v2, timings["encode_v2"], timings["search_v2"] = self.dense_branch("v2", query, dense_text)
        else:
            futures = (pool.submit(self.bm25, query, text),
                       pool.submit(self.dense_branch, "zero", query, dense_text),
                       pool.submit(self.dense_branch, "v2", query, dense_text))
            bm25, plain, timings["bm25"] = futures[0].result()
            zero, timings["encode_zero"], timings["search_zero"] = futures[1].result()
            v2, timings["encode_v2"], timings["search_v2"] = futures[2].result()
        started = time.perf_counter_ns()
        lists = channel_lists(bm25, plain, zero[0], zero[1], v2[0], v2[1])
        frame = query_features(query, str(query.query_id), lists, self.item_side)
        featured = time.perf_counter_ns()
        top50 = ensemble_top50(frame, self.models, self.catboost_threads)[str(query.query_id)]
        timings["features"] = (featured - started) / 1e6
        timings["selector"] = (time.perf_counter_ns() - featured) / 1e6
        assert len(top50) == 50 and len(pool_items(lists)) >= 50
        return top50, timings


def measure(pipeline: Pipeline, queries: pd.DataFrame, warmup: int, parallel: bool):
    pool = ThreadPoolExecutor(max_workers=3) if parallel else None
    warm = pd.concat([queries] * (warmup // len(queries) + 1), ignore_index=True).head(warmup)
    for _, query in warm.iterrows():
        pipeline.run(query, pool)
    totals, per_stage, outputs = [], {name: [] for name in STAGES}, {}
    for _, query in queries.iterrows():
        started = time.perf_counter_ns()
        top50, timings = pipeline.run(query, pool)
        totals.append((time.perf_counter_ns() - started) / 1e6)
        for name in STAGES:
            per_stage[name].append(timings[name])
        outputs[str(query.query_id)] = top50
    if pool is not None:
        pool.shutdown()
    return totals, per_stage, outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objective", choices=("recall50", "yetirank"), default="recall50")
    parser.add_argument("--encoder", choices=("torch", "onnx"), default="torch")
    parser.add_argument("--threads", type=int, default=16, help="intra-op threads per encoder")
    parser.add_argument("--blas-threads", type=int, default=16, help="MKL threads per exact search")
    parser.add_argument("--catboost-threads", type=int, default=-1)
    parser.add_argument("--modes", default="sequential,parallel")
    # The protocol's 25 warm-up queries assume resident indexes. On a laptop
    # whose swap is full, the first few hundred queries after loading ~16 GB
    # still page memory back in; a longer untimed warm-up measures steady state.
    parser.add_argument("--warmup-queries", type=int, default=None)
    parser.add_argument("--gc-freeze", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/latency_final_cpu.json")
    args = parser.parse_args()
    guardrails = json.loads((ROOT / "config/latency_guardrails.json").read_text())
    if args.samples < int(guardrails["research_laptop"]["minimum_measured_queries"]):
        raise ValueError("Guardrail requires more measured queries")
    threshold = float(guardrails["research_laptop"]["warm_p95_ms"])
    warmup = args.warmup_queries or int(guardrails["protocol"]["warmup_queries"])

    pipeline = Pipeline(args.encoder, args.threads, args.seed, args.blas_threads, args.catboost_threads,
                        args.objective)
    if args.gc_freeze:
        # Loading creates millions of long-lived objects (item feature cache,
        # BM25 structures); every full cyclic-GC pass walks all of them and
        # stalls whichever stage is running. Freezing them after loading is
        # the standard serving fix and does not touch any score.
        gc.collect()
        gc.freeze()
    queries = pd.read_parquet(ROOT / "dataset/benchmark_queries.parquet").sample(
        n=args.samples, random_state=args.seed).reset_index(drop=True)

    report = {
        "status": "measured_cpu",
        "pipeline": ("BM25 + zero-shot + LoRA v2 USER-bge-m3, RRF top-200, PU CatBoost selector "
                     f"({'attempt 7, Recall@50 lambda' if args.objective == 'recall50' else 'attempt 6, YetiRankPairwise'}), top-50"),
        "notes": [
            "LoRA v2 searches the real saved passage vectors; zero-shot index is a random unit matrix of the real shape",
            "onnx: real merged LoRA v2 export for the v2 branch; torch: base weights for both (same architecture)",
            f"encoder={args.encoder}, threads per encoder={args.threads}, BLAS threads={args.blas_threads}, "
            f"CatBoost threads={args.catboost_threads}, KMP_BLOCKTIME={os.environ['KMP_BLOCKTIME']}, "
            f"gc.freeze after loading={args.gc_freeze}",
        ],
        "protocol": {**guardrails["protocol"], "warmup_queries_used": warmup},
        "hardware": hardware(), "modes": {},
    }
    outputs = {}
    for mode in args.modes.split(","):
        totals, per_stage, outputs[mode] = measure(pipeline, queries, warmup, parallel=mode == "parallel")
        end_to_end = summarize(totals)
        end_to_end.update(guardrail_p95_ms=threshold, passes_guardrail=bool(end_to_end["p95_ms"] <= threshold))
        report["modes"][mode] = {"stages": {name: summarize(values) for name, values in per_stage.items()},
                                 "end_to_end": end_to_end}
        print(mode, json.dumps({"stages_p95": {k: round(v["p95_ms"], 1) for k, v in report["modes"][mode]["stages"].items()},
                                "end_to_end": {k: round(v, 1) if isinstance(v, float) else v
                                               for k, v in end_to_end.items()}}), flush=True)
    if len(outputs) > 1:
        first, *rest = outputs.values()
        report["identical_top50_across_modes"] = all(other == first for other in rest)
        assert report["identical_top50_across_modes"], "parallel execution must not change the answer"
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", args.output)


if __name__ == "__main__":
    main()
