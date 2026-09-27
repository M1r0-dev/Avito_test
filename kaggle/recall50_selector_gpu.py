# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Stage 25B — Recall@50 objective для CatBoost selector (2×T4)
#
# Текущая система (attempt 6): `BM25 + zero-shot BGE-M3 + LoRA v2 → RRF
# top-200 → PU-bagged selector → top-50`. Каждый из трёх bags (все positives +
# 48 unlabeled, 2/3 hard) обучается `YetiRankPairwise`: он штрафует все
# неправильно упорядоченные пары одинаково, хотя macro Recall@50 зависит
# только от того, попал ли positive в первые 50 из ~175 кандидатов. Здесь
# меняется **только objective внутри bag**; пул, признаки, PU-выборка
# (budget 48, seeds 41–43) и ансамбль по reciprocal rank — как в notebook 24.
#
# Локальная проверка CatBoost 1.2.10 (записана в notebook 25C) показала,
# что встроенного Recall@k objective нет: `LambdaMart`, `StochasticRank` и
# `YetiRank:mode=` отвергают `RecallAt`, а `StochasticFilter` принимает
# `metric=RecallAt;top=50`, но молча игнорирует параметры (FilteredDCG).
# Поэтому прямой objective реализован как **LambdaMART с |ΔRecall@50|**
# поверх CatBoost `PairLogit`:
#
# - каждые `ROUND_TREES` деревьев пересчитываются ранги текущей модели внутри
#   запроса **по полному пулу**, а не по bag: в bag из 49 строк positive всегда
#   в top-50, и Recall@50 там тривиален (tie-break — RRF `fused`);
# - пары — только (positive, sampled unlabeled) из bag: PU решает, каким
#   unlabeled доверять как негативам, лямбда — где проходит cutoff;
# - пара (positive, unlabeled) получает вес `|ΔRecall@50|` от их обмена:
#   `1/|rel_q|`, если они по разные стороны от 50-й позиции, иначе 0
#   (`hard`); `soft` заменяет индикатор `rank<=50` на `σ((50.5−rank)/τ)`;
# - новые деревья обучаются от накопленного score через `baseline` Pool.
#
# Это listwise-градиент именно macro Recall@50: пары внутри top-50 и вне его
# не влияют на метрику и не получают веса, а `1/|rel_q|` — знаменатель метрики.
#
# ## Зафиксированный протокол
#
# - Вход — неизменённый пул notebook 24 (C2 top-200, те же признаки, split);
#   PU budget 48 и seeds 41–43 не перевыбираются.
# - Objectives внутри bag: `YetiRankPairwise` (reference attempt 6), `YetiRank`,
#   `QuerySoftMax` (обычный listwise softmax), `LambdaRecall50_hard`,
#   `LambdaRecall50_soft` (τ=8).
# - CatBoost-параметры: `depth ∈ {6, 8}`, `l2_leaf_reg ∈ {3, 10}`,
#   `learning_rate=0.06`, `border_count=254` (на GPU по умолчанию 128, а
#   ранговые признаки до 501 теряли бы разрешение около cutoff), число деревьев
#   до 1000 с шагом 100 выбирается по staged OOF.
# - OOF — тот же 2-fold cross-fitting на dev, что в notebooks 19/24; recall
#   считается против всех relevant, включая positives вне пула.
# - Выбор: максимум OOF dev Recall@50, tie-break OOF dev-tail, меньше деревьев.
#   Dev (1 226 запросов) вместо dev-tail (~176) — потому что сетка большая, а
#   notebook 24 показал полный tie всех конфигураций на dev-tail.
# - Для лучшей конфигурации каждого objective финальная модель учится на всём
#   dev; test и benchmark top-50 экспортируются. Статистический gate против
#   attempt 4 выполняется локально в notebook 25C, test здесь не оценивается.

# %%
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import json
import os
import queue
import threading
import time

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from catboost.utils import get_gpu_device_count

# Smoke-режим: локальный CPU-прогон на крошечной сетке до отправки на Kaggle.
SMOKE = os.environ.get("RECALL50_SMOKE") == "1"
K = 50
SEED = 42
LEARNING_RATE = 0.06
BORDER_COUNT = 254
MAX_TREES = 100 if SMOKE else 1000
CHECKPOINT = 50 if SMOKE else 100
ROUND_TREES = 25 if SMOKE else 50
GRID_BUDGET_SECONDS = (6 if SMOKE else 62) * 60  # запас квоты на финальные модели
CPU_THREADS_PER_GPU = 2  # Kaggle GPU-сессия даёт 4 vCPU на две карты

LOSSES = {
    "YetiRankPairwise": {"kind": "builtin", "loss_function": "YetiRankPairwise"},
    "YetiRank": {"kind": "builtin", "loss_function": "YetiRank"},
    "QuerySoftMax": {"kind": "builtin", "loss_function": "QuerySoftMax"},
    "LambdaRecall50_hard": {"kind": "lambda", "tau": 0.0},
    "LambdaRecall50_soft": {"kind": "lambda", "tau": 8.0},
    # Вариант, предложенный ранее (Codex): параметры metric/top проба 25C
    # признаёт игнорируемыми. Оставлен для прямого сравнения; CPU-only.
    "StochasticFilter_RecallAt50": {"kind": "builtin", "cpu_only": True,
                                    "loss_function": "StochasticFilter:metric=RecallAt;top=50"},
}
PARAMS = [{"depth": depth, "l2_leaf_reg": l2} for depth in (6, 8) for l2 in (3.0, 10.0)]
# PU-выборка зафиксирована notebook 24 и не перевыбирается.
PU_BUDGET = 48
PU_SEEDS = (41, 42, 43)
HARD_FUSED, HARD_CHANNEL = 75, 40
RR_OFFSET = 20.0
if SMOKE:
    LOSSES = {name: LOSSES[name] for name in ("YetiRankPairwise", "LambdaRecall50_hard", "StochasticFilter_RecallAt50")}
    PARAMS = PARAMS[:1]


def locate_input() -> Path:
    if os.environ.get("RECALL50_INPUT"):
        return Path(os.environ["RECALL50_INPUT"])
    hits = sorted(Path("/kaggle/input").rglob("c2_validation.parquet"))
    if not hits:
        raise FileNotFoundError("private dataset avito-recall50-selector-pool is not attached")
    return hits[0].parent


INPUT = locate_input()
OUTPUT = Path(os.environ.get("RECALL50_OUTPUT", "/kaggle/working"))
OUTPUT.mkdir(parents=True, exist_ok=True)
START = time.time()

# %% [markdown]
# ## Данные и векторизованные ранги
#
# Строки внутри запроса лежат подряд и отсортированы по `fused`, поэтому ранг
# считается одним `lexsort` без groupby — это вызывается на каждом раунде
# лямбд и на каждом staged checkpoint.

# %%
splits = json.loads((INPUT / "splits.json").read_text())
FEATURES = splits["features"]
validation = pd.read_parquet(INPUT / "c2_validation.parquet")
benchmark = pd.read_parquet(INPUT / "c2_benchmark.parquet")
relevant = pd.read_parquet(INPUT / "relevant.parquet")
# Знаменатель macro Recall@50 — все relevant запроса, а не только попавшие в пул.
N_RELEVANT = relevant.groupby("eval_query_id").size()
DEV, TEST, TAIL = set(splits["dev"]), set(splits["test"]), set(splits["tail"])


@dataclass
class Block:
    frame: pd.DataFrame
    X: np.ndarray
    group: np.ndarray
    first_row: np.ndarray
    fused: np.ndarray
    label: np.ndarray | None
    query_ids: np.ndarray
    n_relevant: np.ndarray | None


def make_block(frame: pd.DataFrame, labelled: bool = True) -> Block:
    frame = frame.sort_values(["query_key", "fused"], kind="stable").reset_index(drop=True)
    group, query_ids = pd.factorize(frame.query_key, sort=False)
    first_row = np.r_[0, np.cumsum(np.bincount(group))[:-1]]
    return Block(
        frame=frame, X=frame[FEATURES].to_numpy(np.float32), group=group.astype(np.int32),
        first_row=first_row, fused=frame.fused.to_numpy(), query_ids=np.asarray(query_ids),
        label=frame.label.to_numpy(np.int8) if labelled else None,
        n_relevant=N_RELEVANT.reindex(query_ids).to_numpy(np.float64) if labelled else None,
    )


def within_query_ranks(block: Block, scores: np.ndarray) -> np.ndarray:
    order = np.lexsort((block.fused, -scores, block.group))
    ranks = np.empty(len(scores), dtype=np.int32)
    ranks[order] = np.arange(len(scores)) - block.first_row[block.group[order]] + 1
    return ranks


def recall_sum(block: Block, scores: np.ndarray, ids: set[str]) -> tuple[float, int]:
    """Sum of per-query Recall@50 over `ids`; queries without pool rows count as 0."""
    hits = (block.label == 1) & (within_query_ranks(block, scores) <= K)
    per_query = np.bincount(block.group, weights=hits, minlength=len(block.query_ids)) / block.n_relevant
    values = pd.Series(per_query, index=block.query_ids).reindex(sorted(ids)).fillna(0.0)
    return float(values.sum()), len(values)


def top_lists(block: Block, scores: np.ndarray) -> pd.DataFrame:
    ranks = within_query_ranks(block, scores)
    keep = ranks <= K
    return pd.DataFrame({
        "query_id": block.frame.query_key.to_numpy()[keep], "rank": ranks[keep],
        "item_id": block.frame.item_id.to_numpy()[keep],
    }).sort_values(["query_id", "rank"], kind="stable").reset_index(drop=True)


fold_blocks = []
for fold in splits["cross_folds"]:
    fit_ids, held_ids = set(fold["fit"]), set(fold["held"])
    fold_blocks.append((
        make_block(validation[validation.query_key.isin(fit_ids)]),
        make_block(validation[validation.query_key.isin(held_ids)]),
        held_ids,
    ))
dev_block = make_block(validation[validation.query_key.isin(DEV)])
test_block = make_block(validation[validation.query_key.isin(TEST)])
benchmark_block = make_block(benchmark, labelled=False)
print("rows: dev", len(dev_block.X), "test", len(test_block.X), "benchmark", len(benchmark_block.X))

# %% [markdown]
# ## PU-bagging и objectives
#
# `pu_sample` — порт `pu_sample_best` notebook 24 с тем же порядком вызовов
# RNG. `PUBag` обучает по модели на каждый seed и усредняет `1/(20+rank)`.
# Все модели имеют интерфейс `staged(X)` (score каждые `CHECKPOINT` деревьев)
# и `predict(X, trees)`.

# %%
def subset(block: Block, rows: np.ndarray) -> Block:
    # rows отсортированы, поэтому порядок (query_key, fused) сохраняется и
    # строка i bag соответствует строке rows[i] полного пула.
    return make_block(block.frame.iloc[rows])


def pu_sample(block: Block, seed: int) -> np.ndarray:
    """All positives + PU_BUDGET unlabeled per query, 2/3 of them hard (notebook 24)."""
    rng = np.random.default_rng(seed)
    hard_budget = int(round(PU_BUDGET * 2 / 3))
    min_channel = block.frame[["bm25", "zero", "v2"]].min(axis=1).to_numpy()
    stops = np.r_[block.first_row[1:], len(block.X)]
    chosen_rows = []
    for start, stop in zip(block.first_row, stops):
        rows = np.arange(start, stop)
        positive = rows[block.label[rows] == 1]
        negative = rows[block.label[rows] == 0]
        is_hard = (block.fused[negative] <= HARD_FUSED) | (min_channel[negative] <= HARD_CHANNEL)
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


def catboost_kwargs(loss_function: str, params: dict, iterations: int,
                    device: int | None, seed: int) -> dict:
    kwargs = dict(
        loss_function=loss_function, iterations=iterations, learning_rate=LEARNING_RATE,
        depth=params["depth"], l2_leaf_reg=params["l2_leaf_reg"], border_count=BORDER_COUNT,
        random_seed=seed, verbose=False, allow_writing_files=False,
    )
    if device is None:
        kwargs.update(task_type="CPU", thread_count=-1)
    else:
        kwargs.update(task_type="GPU", devices=str(device), thread_count=CPU_THREADS_PER_GPU)
    return kwargs


class Builtin:
    def __init__(self, loss_function: str, params: dict, device: int | None, seed: int):
        self.loss_function, self.params, self.device, self.seed = loss_function, params, device, seed

    def fit(self, block: Block, rows: np.ndarray, trees: int) -> "Builtin":
        bag = subset(block, rows)
        self.model = CatBoostRanker(**catboost_kwargs(
            self.loss_function, self.params, trees, self.device, self.seed))
        self.model.fit(Pool(bag.X, label=bag.label, group_id=bag.group))
        return self

    def staged(self, X: np.ndarray):
        for step, scores in enumerate(self.model.staged_predict(X, eval_period=CHECKPOINT), 1):
            yield step * CHECKPOINT, scores

    def predict(self, X: np.ndarray, trees: int) -> np.ndarray:
        return self.model.predict(X, ntree_end=trees)


def positive_pairs(block: Block) -> np.ndarray:
    """All (positive, unlabeled) row pairs inside each query of `block`."""
    stops = np.r_[block.first_row[1:], len(block.X)]
    pairs = []
    for row in np.flatnonzero(block.label == 1):
        start, stop = block.first_row[block.group[row]], stops[block.group[row]]
        negatives = start + np.flatnonzero(block.label[start:stop] == 0)
        pairs.append(np.column_stack([np.full(len(negatives), row), negatives]))
    return np.concatenate(pairs).astype(np.int64)


class LambdaRecall:
    """LambdaMART with |ΔRecall@50| weights on top of CatBoost PairLogit."""

    def __init__(self, tau: float, params: dict, device: int | None, seed: int):
        self.tau, self.params, self.device, self.seed = tau, params, device, seed

    def inside_top(self, ranks: np.ndarray) -> np.ndarray:
        if self.tau == 0:
            return (ranks <= K).astype(np.float64)
        return 1.0 / (1.0 + np.exp(-(K + 0.5 - ranks) / self.tau))

    def fit(self, block: Block, rows: np.ndarray, trees: int) -> "LambdaRecall":
        bag = subset(block, rows)
        pairs = positive_pairs(bag)
        # Score ведётся на полном пуле: cutoff 50 существует только там.
        # Нулевой старт → первый раунд ранжирует по RRF `fused` (tie-break).
        scores = np.zeros(len(block.X))
        self.models = []
        for _ in range(trees // ROUND_TREES):
            inside = self.inside_top(within_query_ranks(block, scores))[rows]
            weights = np.abs(inside[pairs[:, 0]] - inside[pairs[:, 1]]) / bag.n_relevant[bag.group[pairs[:, 0]]]
            keep = weights > 1e-6
            model = CatBoostRanker(**catboost_kwargs(
                "PairLogit", self.params, ROUND_TREES, self.device, self.seed))
            model.fit(Pool(bag.X, group_id=bag.group, pairs=pairs[keep],
                           pairs_weight=weights[keep], baseline=scores[rows]))
            scores = scores + model.predict(block.X)
            self.models.append(model)
        return self

    def staged(self, X: np.ndarray):
        total = np.zeros(len(X))
        for index, model in enumerate(self.models, 1):
            total = total + model.predict(X)
            if (index * ROUND_TREES) % CHECKPOINT == 0:
                yield index * ROUND_TREES, total.copy()

    def predict(self, X: np.ndarray, trees: int) -> np.ndarray:
        return np.sum([model.predict(X) for model in self.models[:trees // ROUND_TREES]], axis=0)


def rank_ensemble(block: Block, member_scores: list[np.ndarray]) -> np.ndarray:
    # Шкалы CatBoost у bag-моделей разные, поэтому усредняются ранги (notebook 24).
    return np.mean([1.0 / (RR_OFFSET + within_query_ranks(block, s)) for s in member_scores], axis=0)


class PUBag:
    def __init__(self, loss: str, params: dict, device: int | None):
        self.loss, self.params, self.device = loss, params, device

    def fit(self, block: Block, trees: int) -> "PUBag":
        spec = LOSSES[self.loss]
        self.members = []
        for seed in PU_SEEDS:
            member = (LambdaRecall(spec["tau"], self.params, self.device, seed)
                      if spec["kind"] == "lambda" else
                      Builtin(spec["loss_function"], self.params, self.device, seed))
            self.members.append(member.fit(block, pu_sample(block, seed), trees))
        return self

    def staged(self, block: Block):
        for checkpoint in zip(*(member.staged(block.X) for member in self.members)):
            yield checkpoint[0][0], rank_ensemble(block, [scores for _, scores in checkpoint])

    def predict(self, block: Block, trees: int) -> np.ndarray:
        return rank_ensemble(block, [member.predict(block.X, trees) for member in self.members])


# %% [markdown]
# ## GPU-проверка и распределение по двум T4
#
# `PairLogit` с явными парами, их весами и baseline на GPU — не самый
# частый режим CatBoost, поэтому он проверяется до сетки. При отказе
# лямбда-objective переводится на CPU и это записывается в `run.json`.

# %%
GPU_COUNT = 0 if SMOKE else get_gpu_device_count()
DEVICES = list(range(GPU_COUNT)) or [None]
LAMBDA_ON_GPU = False
LAMBDA_ROUND_SECONDS: dict[str, float] = {}
if GPU_COUNT:
    # Один раунд = отдельный fit; на малых bags инициализация GPU может стоить
    # дороже самих 50 деревьев, поэтому устройство выбирается по замеру.
    probe_block = fold_blocks[0][0]
    probe_rows = pu_sample(probe_block, PU_SEEDS[0])
    for name, device in (("cpu", None), ("gpu", 0)):
        started = time.time()
        try:
            LambdaRecall(0.0, PARAMS[0], device, PU_SEEDS[0]).fit(probe_block, probe_rows, 2 * ROUND_TREES)
            LAMBDA_ROUND_SECONDS[name] = (time.time() - started) / 2
        except Exception as error:  # noqa: BLE001 — любая причина отказа фиксируется
            print(f"PairLogit with pair weights failed on {name}:", repr(error)[:300])
    LAMBDA_ON_GPU = "gpu" in LAMBDA_ROUND_SECONDS and (
        LAMBDA_ROUND_SECONDS["gpu"] < LAMBDA_ROUND_SECONDS.get("cpu", float("inf")))
print({"catboost": catboost.__version__, "gpus": GPU_COUNT, "lambda_on_gpu": LAMBDA_ON_GPU,
       "lambda_round_seconds": LAMBDA_ROUND_SECONDS})

def job_device(loss: str, device: int | None) -> int | None:
    spec = LOSSES[loss]
    if spec.get("cpu_only") or (spec["kind"] == "lambda" and not LAMBDA_ON_GPU):
        return None
    return device


device_pool: queue.Queue = queue.Queue()
for device in DEVICES:
    device_pool.put(device)
lock = threading.Lock()
grid_rows: list[dict] = []
job_log: list[dict] = []


def run_job(job: tuple[str, int, int]) -> None:
    loss, param_index, fold = job
    if time.time() - START > GRID_BUDGET_SECONDS:
        with lock:
            job_log.append({"loss": loss, "param_index": param_index, "fold": fold, "status": "skipped_budget"})
        return
    device = device_pool.get()
    try:
        run_device = job_device(loss, device)
        params = PARAMS[param_index]
        fit_block, held_block, held_ids = fold_blocks[fold]
        started = time.time()
        try:
            model = PUBag(loss, params, run_device).fit(fit_block, MAX_TREES)
        except Exception as error:  # noqa: BLE001 — отказ одного objective не должен ронять сетку
            with lock:
                job_log.append({"loss": loss, "param_index": param_index, "fold": fold,
                                "device": run_device, "status": "failed", "error": repr(error)[:300]})
            print("FAILED", loss, params, fold, repr(error)[:300], flush=True)
            return
        rows = []
        for trees, scores in model.staged(held_block):
            for segment, ids in {"dev": held_ids, "dev_tail": held_ids & TAIL}.items():
                total, count = recall_sum(held_block, scores, ids)
                rows.append({"loss": loss, **params, "fold": fold, "trees": trees,
                             "segment": segment, "recall_sum": total, "n": count})
        seconds = time.time() - started
        with lock:
            grid_rows.extend(rows)
            job_log.append({"loss": loss, "param_index": param_index, "fold": fold,
                            "device": run_device, "seconds": seconds, "status": "done"})
            # Инкрементальная запись: если квота кончится, частичная сетка сохранится.
            pd.DataFrame(grid_rows).to_parquet(OUTPUT / "grid.parquet", index=False)
        best = max(r["recall_sum"] / r["n"] for r in rows if r["segment"] == "dev")
        print(f"[{time.time() - START:6.0f}s] {loss} {params} fold={fold} gpu={run_device} "
              f"{seconds:.0f}s best_held={best:.4f}", flush=True)
    finally:
        device_pool.put(device)


# Порядок задаёт приоритет при нехватке времени: сначала все objectives на
# базовых параметрах, затем расширение сетки.
jobs = [(loss, p, fold) for p in range(len(PARAMS)) for loss in LOSSES for fold in range(len(fold_blocks))]
with ThreadPoolExecutor(max_workers=len(DEVICES)) as pool:
    list(pool.map(run_job, jobs))

# %% [markdown]
# ## Выбор по OOF dev
#
# Конфигурация допускается к выбору, только если посчитаны оба fold'а: иначе
# OOF-оценка покрывала бы половину dev.

# %%
grid = pd.DataFrame(grid_rows)
keys = ["loss", "depth", "l2_leaf_reg", "trees"]
summary = grid.groupby([*keys, "segment"]).agg(
    recall_sum=("recall_sum", "sum"), n=("n", "sum"), folds=("fold", "nunique")).reset_index()
summary = summary[summary.folds.eq(len(fold_blocks))]
summary["recall"] = summary.recall_sum / summary.n
selection = summary.pivot_table(index=keys, columns="segment", values="recall").reset_index()
selection["loss_order"] = selection.loss.map({name: i for i, name in enumerate(LOSSES)})
selection = selection.sort_values(
    ["dev", "dev_tail", "trees", "loss_order"], ascending=[False, False, True, True], kind="stable"
).drop(columns="loss_order").reset_index(drop=True)
selection.to_parquet(OUTPUT / "selection.parquet", index=False)
best_per_loss = selection.groupby("loss", sort=False).head(1).reset_index(drop=True)
print(best_per_loss.round(5).to_string())
selected = best_per_loss.iloc[0].to_dict()
print("selected:", selected)

# %% [markdown]
# ## Финальные модели на всём dev
#
# Для каждого objective — его лучшая конфигурация; test и benchmark top-50
# экспортируются для локального gate (25C) и описательной абляции лоссов.

# %%
test_parts, benchmark_parts = [], []


def final_job(row: dict) -> None:
    device = device_pool.get()
    try:
        loss = row["loss"]
        run_device = job_device(loss, device)
        params = {"depth": int(row["depth"]), "l2_leaf_reg": float(row["l2_leaf_reg"])}
        trees = int(row["trees"])
        try:
            model = PUBag(loss, params, run_device).fit(dev_block, trees)
        except Exception as error:  # noqa: BLE001
            print("FAILED final", loss, repr(error)[:300], flush=True)
            return
        test_top = top_lists(test_block, model.predict(test_block, trees)).assign(loss=loss)
        benchmark_top = top_lists(benchmark_block, model.predict(benchmark_block, trees)).assign(loss=loss)
        with lock:
            test_parts.append(test_top)
            benchmark_parts.append(benchmark_top)
        print(f"[{time.time() - START:6.0f}s] final {loss} {params} trees={trees}", flush=True)
    finally:
        device_pool.put(device)


with ThreadPoolExecutor(max_workers=len(DEVICES)) as pool:
    list(pool.map(final_job, best_per_loss.to_dict("records")))

pd.concat(test_parts, ignore_index=True).to_parquet(OUTPUT / "test_top50.parquet", index=False)
pd.concat(benchmark_parts, ignore_index=True).to_parquet(OUTPUT / "benchmark_top50.parquet", index=False)
(OUTPUT / "run.json").write_text(json.dumps({
    "catboost": catboost.__version__, "gpus": GPU_COUNT, "lambda_on_gpu": LAMBDA_ON_GPU,
    "lambda_round_seconds": LAMBDA_ROUND_SECONDS, "smoke": SMOKE, "learning_rate": LEARNING_RATE, "border_count": BORDER_COUNT,
    "pu_budget": PU_BUDGET, "pu_seeds": PU_SEEDS, "max_trees": MAX_TREES, "checkpoint": CHECKPOINT, "round_trees": ROUND_TREES,
    "losses": LOSSES, "params": PARAMS, "selected": selected,
    "best_per_loss": best_per_loss.to_dict("records"), "jobs": job_log,
    "seconds": time.time() - START,
}, indent=2, default=str), encoding="utf-8")
print("done in", round(time.time() - START), "s")
