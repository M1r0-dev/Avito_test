#!/usr/bin/env python
"""Measure reproducible warm batch=1 latency for implemented retrieval stages."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avito_retrieval.bm25 import SparseBM25  # noqa: E402
from avito_retrieval.channels import filtered_ranking_from_scores  # noqa: E402
from avito_retrieval.text import query_text  # noqa: E402


def hardware() -> dict[str, object]:
    try:
        cpu = subprocess.run(
            ["lscpu", "-J"], check=True, capture_output=True, text=True
        ).stdout
        cpu_info: object = json.loads(cpu)
    except (FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError):
        cpu_info = platform.processor()
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        ).stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.SubprocessError):
        gpu = []
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu_info,
        "gpu": gpu,
    }


def summarize(values: list[float]) -> dict[str, float | int]:
    timings = np.asarray(values, dtype=np.float64)
    return {
        "n": len(timings),
        "mean_ms": float(timings.mean()),
        "p50_ms": float(np.quantile(timings, 0.50)),
        "p95_ms": float(np.quantile(timings, 0.95)),
        "p99_ms": float(np.quantile(timings, 0.99)),
        "max_ms": float(timings.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/latency_bm25.json")
    args = parser.parse_args()

    guardrails = json.loads((ROOT / "config/latency_guardrails.json").read_text())
    required = int(guardrails["research_laptop"]["minimum_measured_queries"])
    if args.samples < required:
        raise ValueError(f"Guardrail requires at least {required} measured queries")

    index = SparseBM25.load(ROOT / "artifacts/bm25")
    items = pd.read_parquet(ROOT / "artifacts/bm25/items.parquet")
    queries = pd.read_parquet(ROOT / "dataset/benchmark_queries.parquet").sample(
        n=args.samples, random_state=args.seed
    ).reset_index(drop=True)

    def retrieve(query: pd.Series) -> list[str]:
        # One sparse multiplication is reused by global and local filters.
        scores = index.score(query_text(query))
        return filtered_ranking_from_scores(
            query, items, scores, retrieve_k=250, output_k=250
        )

    warmup = int(guardrails["protocol"]["warmup_queries"])
    for _, query in queries.head(warmup).iterrows():
        retrieve(query)
    durations = []
    for _, query in queries.iterrows():
        started = time.perf_counter_ns()
        result = retrieve(query)
        durations.append((time.perf_counter_ns() - started) / 1_000_000)
        if len(result) != 250:
            raise AssertionError("BM25 latency path must still return top-250")

    stage = summarize(durations)
    threshold = float(guardrails["research_laptop"]["warm_p95_ms"])
    stage["guardrail_p95_ms"] = threshold
    stage["passes_guardrail"] = bool(stage["p95_ms"] <= threshold)
    report = {
        "status": "partial",
        "reason": "BM25 measured locally; GPU dense and full accepted LTR pipeline pending",
        "protocol": guardrails["protocol"],
        "hardware": hardware(),
        "stages": {"bm25_one_pass": stage},
        "end_to_end": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not stage["passes_guardrail"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
