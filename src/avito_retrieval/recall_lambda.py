"""Recall@50 lambda objective of the final selector (public attempt 7).

CatBoost has no Recall@k objective (notebook 25C: `RecallAt` is rejected by
LambdaMart/StochasticRank/YetiRank and ignored by StochasticFilter). The
selector therefore trains LambdaMART with |ΔRecall@50| pair weights on top of
CatBoost `PairLogit`: every `ROUND_TREES` trees the current within-query ranks
of the full pool are recomputed, a (positive, sampled unlabeled) pair gets
weight |π(r_pos) − π(r_neg)| / |relevant_q| with π(r) = σ((50.5 − r)/τ), and
the next trees continue from the accumulated score through the Pool baseline.

This is a port of `kaggle/recall50_selector_gpu.py` (stage 25B, run on CPU as
`notebooks/25b_recall50_lambda_cpu.ipynb`) restricted to the selected
configuration; `scripts/train_pu_selector.py --objective recall50` retrains it
and `scripts/generate_answer.py --check` verifies the submitted attempt 7.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

from .pu_selector import FEATURES, HARD_CHANNEL, HARD_FUSED, PU_BUDGET, PU_SEEDS, SOURCES

# Selected on 2-fold OOF dev in stage 25B (notebook 25C): τ=8, cutoff 50,
# depth 6, l2 3, 100 trees = two rounds of 50.
K = 50
TAU = 8.0
ROUND_TREES = 50
TREES = 100
LEARNING_RATE = 0.06
DEPTH = 6
L2_LEAF_REG = 3.0
BORDER_COUNT = 254


def _groups(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Group codes and first row of each query; `frame` sorted by (query_key, fused)."""
    group = pd.factorize(frame.query_key, sort=False)[0].astype(np.int32)
    first_row = np.r_[0, np.cumsum(np.bincount(group))[:-1]]
    return group, first_row


def within_query_ranks(group: np.ndarray, first_row: np.ndarray, fused: np.ndarray,
                       scores: np.ndarray) -> np.ndarray:
    """1-based rank by score inside each query, ties broken by the RRF `fused` rank."""
    order = np.lexsort((fused, -scores, group))
    ranks = np.empty(len(scores), dtype=np.int32)
    ranks[order] = np.arange(len(scores)) - first_row[group[order]] + 1
    return ranks


def pu_rows(frame: pd.DataFrame, seed: int) -> np.ndarray:
    """Sorted row positions of one PU bag: all positives + PU_BUDGET unlabeled per query.

    Same RNG calls, in the same order, as notebook 24 `pu_sample_best` and the
    stage-25B kernel, so the bag is identical.
    """
    rng = np.random.default_rng(seed)
    group, first_row = _groups(frame)
    label = frame.label.to_numpy()
    fused = frame.fused.to_numpy()
    min_channel = frame[list(SOURCES)].min(axis=1).to_numpy()
    hard_budget = int(round(PU_BUDGET * 2 / 3))
    stops = np.r_[first_row[1:], len(frame)]
    chosen_rows = []
    for start, stop in zip(first_row, stops):
        rows = np.arange(start, stop)
        positive = rows[label[rows] == 1]
        negative = rows[label[rows] == 0]
        is_hard = (fused[negative] <= HARD_FUSED) | (min_channel[negative] <= HARD_CHANNEL)
        hard, other = negative[is_hard], negative[~is_hard]
        take_hard = min(hard_budget, len(hard))
        hard_idx = rng.choice(hard, size=take_hard, replace=False)
        take_other = min(PU_BUDGET - take_hard, len(other))
        other_idx = rng.choice(other, size=take_other, replace=False)
        chosen = set(hard_idx.tolist()) | set(other_idx.tolist())
        missing = min(PU_BUDGET - len(chosen), len(negative) - len(chosen))
        if missing > 0:
            available = negative[~np.isin(negative, list(chosen))]
            chosen.update(rng.choice(available, size=missing, replace=False).tolist())
        chosen_rows.append(np.concatenate([positive, np.asarray(sorted(chosen), dtype=np.int64)]))
    return np.sort(np.concatenate(chosen_rows))


def _positive_pairs(label: np.ndarray, group: np.ndarray, first_row: np.ndarray) -> np.ndarray:
    stops = np.r_[first_row[1:], len(label)]
    pairs = []
    for row in np.flatnonzero(label == 1):
        start, stop = first_row[group[row]], stops[group[row]]
        negatives = start + np.flatnonzero(label[start:stop] == 0)
        pairs.append(np.column_stack([np.full(len(negatives), row), negatives]))
    return np.concatenate(pairs).astype(np.int64)


def train_bag(frame: pd.DataFrame, n_relevant: Mapping[str, int], seed: int) -> list[CatBoostRanker]:
    """Round models of one PU bag; their raw predictions sum to the bag score."""
    group, first_row = _groups(frame)
    fused = frame.fused.to_numpy()
    X = frame[FEATURES].to_numpy(np.float32)
    rows = pu_rows(frame, seed)
    bag = frame.iloc[rows]
    bag_group, bag_first = _groups(bag)
    bag_label = bag.label.to_numpy(np.int8)
    bag_relevant = bag.query_key.map(n_relevant).to_numpy(np.float64)
    pairs = _positive_pairs(bag_label, bag_group, bag_first)
    # Scores live on the full pool: the 50 cutoff only exists there (a bag has
    # ~49 rows, so every positive would trivially be in its top 50).
    scores = np.zeros(len(frame))
    models = []
    for _ in range(TREES // ROUND_TREES):
        ranks = within_query_ranks(group, first_row, fused, scores)
        inside = (1.0 / (1.0 + np.exp(-(K + 0.5 - ranks) / TAU)))[rows]
        weights = np.abs(inside[pairs[:, 0]] - inside[pairs[:, 1]]) / bag_relevant[pairs[:, 0]]
        keep = weights > 1e-6
        model = CatBoostRanker(loss_function="PairLogit", iterations=ROUND_TREES,
                               learning_rate=LEARNING_RATE, depth=DEPTH, l2_leaf_reg=L2_LEAF_REG,
                               border_count=BORDER_COUNT, random_seed=seed, verbose=False,
                               allow_writing_files=False, task_type="CPU", thread_count=-1)
        model.fit(Pool(X[rows], group_id=bag_group, pairs=pairs[keep],
                       pairs_weight=weights[keep], baseline=scores[rows]))
        scores = scores + model.predict(X)
        models.append(model)
    return models


class SummedRanker:
    """Score of one bag = sum of its round models' raw predictions."""

    def __init__(self, models: Sequence[CatBoostRanker]) -> None:
        self.models = list(models)

    def predict(self, features: pd.DataFrame, thread_count: int = -1) -> np.ndarray:
        return np.sum([model.predict(features, thread_count=thread_count) for model in self.models], axis=0)


def train_models(frame: pd.DataFrame, n_relevant: Mapping[str, int],
                 seeds: Sequence[int] = PU_SEEDS) -> list[list[CatBoostRanker]]:
    """All bags; `frame` sorted by (query_key, fused) with a `label` column."""
    return [train_bag(frame, n_relevant, seed) for seed in seeds]
