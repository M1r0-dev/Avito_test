#!/usr/bin/env python
"""Warm batch=1 CPU latency of the full attempt-2 candidate generator.

The guardrail (`config/latency_guardrails.json`) was only measured for BM25
(`reports/latency_bm25.json`, status "partial"). Attempt 2 also encodes every
query with two USER-bge-m3 encoders (zero-shot and fine-tuned) and runs exact
global + location-local passage search for each. This script times that whole
online path per query:

    normalize -> BM25 global/local -> 2 x (encode query -> global search ->
    local search -> passage-to-item collapse with category filter ->
    global/local RRF) -> outer RRF -> top-50

Why the passage matrices are synthetic: the Kaggle kernels saved rankings, not
embeddings, and re-encoding 554,920 passages twice on CPU takes many hours.
Exact inner-product search costs the same for any values of a float32
matrix of the same shape, so random unit vectors give the same timing. The
passage-to-item map and location sizes are real (built with the same
chunking as the kernels). The fine-tuned encoder has the base architecture, so
the second encoder is timed with base weights. Rankings produced here are
meaningless and only their sizes are checked.

Passages are stored sorted by item location, so the local channel is a
contiguous slice (no gather), which is how an online index would store them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from avito_retrieval.bm25 import SparseBM25  # noqa: E402
from avito_retrieval.channels import filtered_ranking_from_scores  # noqa: E402
from avito_retrieval.fusion import reciprocal_rank_fusion  # noqa: E402
from avito_retrieval.text import item_passages, query_text  # noqa: E402
from benchmark_latency import hardware, summarize  # noqa: E402

MODEL = "deepvk/USER-bge-m3"
MODEL_REVISION = "0cc6cfe48e260fb0474c753087a69369e88709ae"
DIM = 1024
# Same depths as the Kaggle kernels: global top-3000 passages (2 shards x
# 1500), local top-750 passages, collapsed to at most 250 items.
GLOBAL_PASSAGES = 3000
LOCAL_PASSAGES = 750
TOP_ITEMS = 250
ATTEMPT_2_WEIGHTS = [1.0, 0.75, 1.25]


class DenseChannel:
    """Exact dense search over location-sorted passage embeddings."""

    def __init__(self, embeddings: np.ndarray, passage_items: np.ndarray,
                 slices: dict[int, slice], categories: np.ndarray, item_ids: np.ndarray,
                 local_weight: float) -> None:
        self.embeddings, self.passage_items, self.slices = embeddings, passage_items, slices
        self.categories, self.item_ids, self.local_weight = categories, item_ids, local_weight

    def collapse(self, passages: np.ndarray, category: int) -> list[str]:
        answer, seen = [], set()
        for item_row in self.passage_items[passages]:
            if category and self.categories[item_row] != category:
                continue
            if item_row not in seen:
                seen.add(item_row)
                answer.append(self.item_ids[item_row])
                if len(answer) == TOP_ITEMS:
                    break
        return answer

    def search(self, vector: np.ndarray, category: int, location: int) -> list[str]:
        scores = self.embeddings @ vector
        top = np.argpartition(scores, -GLOBAL_PASSAGES)[-GLOBAL_PASSAGES:]
        global_rank = self.collapse(top[np.argsort(scores[top])[::-1]], category)
        rankings, weights = [global_rank], [1.0]
        part = self.slices.get(location)
        if part is not None:
            local_scores = self.embeddings[part] @ vector
            k = min(LOCAL_PASSAGES, len(local_scores))
            local_top = np.argpartition(local_scores, -k)[-k:]
            local_top = local_top[np.argsort(local_scores[local_top])[::-1]] + part.start
            rankings.append(self.collapse(local_top, category))
            weights.append(self.local_weight)
        return reciprocal_rank_fusion(rankings, weights=weights, rrf_k=60, top_k=TOP_ITEMS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/latency_attempt2_cpu.json")
    args = parser.parse_args()
    guardrails = json.loads((ROOT / "config/latency_guardrails.json").read_text())
    if args.samples < int(guardrails["research_laptop"]["minimum_measured_queries"]):
        raise ValueError("Guardrail requires more measured queries")

    # --- offline part (excluded from timing by the protocol) ---------------
    index = SparseBM25.load(ROOT / "artifacts/bm25")
    items = pd.read_parquet(ROOT / "artifacts/bm25/items.parquet")
    raw = pd.read_parquet(ROOT / "dataset/benchmark_items.parquet",
                          columns=["item_id", "item_title_raw", "item_infm_params_text",
                                   "item_description_raw"])
    assert (raw.item_id.values == items.item_id.values).all()
    chunks = np.array([sum(1 for _ in item_passages(row)) for row in raw.to_dict("records")])
    passage_items = np.repeat(np.arange(len(items)), chunks)
    locations = items.item_location_id.to_numpy()[passage_items]
    order = np.argsort(locations, kind="stable")
    passage_items, locations = passage_items[order], locations[order]
    boundaries = np.flatnonzero(np.diff(locations)) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(locations)]
    slices = {int(locations[s]): slice(int(s), int(e)) for s, e in zip(starts, ends)}
    print(f"passages={len(passage_items):,}; locations={len(slices):,}", flush=True)

    rng = np.random.default_rng(args.seed)
    categories = items.item_category_id.to_numpy()
    item_ids = items.item_id.astype(str).to_numpy()
    channels = []
    for local_weight in (1.15, 1.5):  # zero-shot, fine-tuned (notebooks 05, 08)
        matrix = rng.standard_normal((len(passage_items), DIM), dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        channels.append(DenseChannel(matrix, passage_items, slices, categories, item_ids, local_weight))
    encoders = [SentenceTransformer(MODEL, revision=MODEL_REVISION, device="cpu") for _ in range(2)]

    queries = pd.read_parquet(ROOT / "dataset/benchmark_queries.parquet").sample(
        n=args.samples, random_state=args.seed).reset_index(drop=True)

    stage_names = ["bm25", "encode_zero", "search_zero", "encode_fine", "search_fine", "fusion"]

    def run(query: pd.Series) -> tuple[list[str], dict[str, float]]:
        timings, clock = {}, time.perf_counter_ns()

        def lap(name: str) -> None:
            nonlocal clock
            now = time.perf_counter_ns()
            timings[name] = (now - clock) / 1e6
            clock = now

        text = query_text(query)
        bm25 = filtered_ranking_from_scores(query, items, index.score(text), retrieve_k=250, output_k=250)
        lap("bm25")
        dense = []
        for name, encoder, channel in zip(("zero", "fine"), encoders, channels):
            with torch.inference_mode():
                vector = encoder.encode(text, normalize_embeddings=True, convert_to_numpy=True)
            lap(f"encode_{name}")
            dense.append(channel.search(vector.astype(np.float32), int(query.search_category),
                                        int(query.search_location_id)))
            lap(f"search_{name}")
        top50 = reciprocal_rank_fusion([bm25, *dense], weights=ATTEMPT_2_WEIGHTS, rrf_k=20, top_k=50)
        lap("fusion")
        return top50, timings

    for _, query in queries.head(int(guardrails["protocol"]["warmup_queries"])).iterrows():
        run(query)
    totals, per_stage = [], {name: [] for name in stage_names}
    for _, query in queries.iterrows():
        started = time.perf_counter_ns()
        top50, timings = run(query)
        totals.append((time.perf_counter_ns() - started) / 1e6)
        for name in stage_names:
            per_stage[name].append(timings[name])
        assert 0 < len(top50) <= 50

    threshold = float(guardrails["research_laptop"]["warm_p95_ms"])
    end_to_end = summarize(totals)
    end_to_end.update(guardrail_p95_ms=threshold, passes_guardrail=bool(end_to_end["p95_ms"] <= threshold))
    report = {
        "status": "measured_cpu",
        "pipeline": "attempt 2: BM25 + zero-shot USER-bge-m3 + fine-tuned USER-bge-m3, RRF top-50",
        "notes": [
            "passage embeddings are synthetic unit vectors of the real shape; exact search cost is value-independent",
            "fine-tuned encoder timed with base weights (identical architecture)",
            f"torch threads={torch.get_num_threads()}",
        ],
        "protocol": guardrails["protocol"],
        "hardware": hardware(),
        "stages": {name: summarize(values) for name, values in per_stage.items()},
        "end_to_end": end_to_end,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stages": {k: round(v["p95_ms"], 1) for k, v in report["stages"].items()},
                      "end_to_end": end_to_end}, indent=2))


if __name__ == "__main__":
    main()
