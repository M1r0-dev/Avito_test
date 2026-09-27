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
# # 25C. Recall@50 objective в PU-bagged selector: статистический gate
#
# Система attempt 6: `BM25 + zero-shot BGE-M3 + LoRA v2 → RRF top-200 →
# PU-bagged selector → top-50`. Stage 25B заменяет objective внутри PU bags.
#
# Ход stage 25B: Kaggle v1 (2×T4) остановлен вручную через ~20 минут, успев
# посчитать все objectives только на первом наборе параметров (depth 6, l2 3);
# его output не сохранился, поэтому OOF-оценки ниже перенесены из лога. По
# ним выбран `LambdaRecall50_soft` — делит первое место с `QuerySoftMax`,
# но в 10 раз быстрее. Он досчитан локально на CPU тем же кодом
# (`notebooks/25b_recall50_lambda_cpu.ipynb`, `RECALL50_ONLY_LOSS`): staged OOF
# выбирает число деревьев, финальная модель учится на dev. Здесь:
#
# 1. фиксируется, почему objective собственный, а не встроенный CatBoost;
# 2. разбирается OOF-выбор 25B;
# 3. **одна** test-проверка выбранной конфигурации против attempt 4 (лучший
#    public; attempt 6 совпадает с ним на test) — test-tail primary, test
#    secondary, `alpha=.025` на endpoint, как в notebook 24;
# 4. описательная абляция: лучший вариант каждого objective против
#    `YetiRankPairwise` той же PU-схемы, плюс прямое сравнение LambdaRecall50
#    с ранее предложенным `StochasticFilter:metric=RecallAt;top=50`
#    (решения по ним не принимаются).
#
# Notebook 12 уже сравнивал встроенные `PairLogitPairwise`, `YetiRank` и
# `QuerySoftMax` на старом LTR-пуле; objective, напрямую завязанного на
# cutoff 50, там не было.
#
# `answer.csv` заменяется только при неотрицательном test-tail и значимом
# primary или secondary endpoint.

# %%
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
import catboost

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from avito_retrieval.metrics import recall_at_k
from avito_retrieval.statistics import paired_recall_test, per_query_recall

INPUT = ROOT / "artifacts/recall50_kaggle_input"   # stage 25A
STAGE = ROOT / "artifacts/recall50_stage25b"        # output stage 25B (CPU)
RANDOM_SEED = 42
ALPHA = 0.025
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def select(mapping: dict, ids: set[str]) -> dict:
    return {query_id: mapping[query_id] for query_id in ids}


def lists(frame: pd.DataFrame) -> dict[str, list[str]]:
    ordered = frame.sort_values(["query_id", "rank"], kind="stable")
    return ordered.groupby("query_id", sort=False).item_id.agg(list).to_dict()

# %% [markdown]
# ## 1. Какие Recall@50 objectives есть в CatBoost
#
# Синтетическая проба (150 кандидатов на запрос, как у пула) проверяет
# каждый кандидатный objective. Параметр считается принятым, только если он
# меняет модель: CatBoost иногда принимает неизвестные параметры без ошибки.

# %%
rng = np.random.default_rng(0)
groups, per_group = 200, 150
X = rng.normal(size=(groups * per_group, 8))
y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=len(X)) > 2.2).astype(int)
probe_pool = Pool(X, label=y, group_id=np.repeat(np.arange(groups), per_group))


def probe(loss: str) -> np.ndarray | str:
    try:
        model = CatBoostRanker(loss_function=loss, iterations=30, depth=4, random_seed=1,
                               verbose=False, allow_writing_files=False, thread_count=8)
        model.fit(probe_pool)
        return model.predict(X)
    except Exception as error:  # noqa: BLE001 — текст отказа и есть результат
        return str(error).splitlines()[0]


probe_results = {loss: probe(loss) for loss in (
    "StochasticRank:metric=RecallAt;top=50", "LambdaMart:metric=RecallAt;top=50",
    "YetiRank:mode=RecallAt;top=50", "StochasticFilter:metric=RecallAt;top=50",
    "StochasticFilter", "StochasticFilter:metric=NDCG;top=50",
)}
probe_table = pd.DataFrame([{
    "objective": loss,
    "accepted": not isinstance(result, str),
    "error": result if isinstance(result, str) else "",
} for loss, result in probe_results.items()])
stochastic_filter_ignores_metric = bool(
    np.allclose(probe_results["StochasticFilter:metric=RecallAt;top=50"], probe_results["StochasticFilter"])
    and np.allclose(probe_results["StochasticFilter:metric=NDCG;top=50"], probe_results["StochasticFilter"])
)
display(probe_table)
print({"catboost": catboost.__version__,
       "StochasticFilter ignores metric/top": stochastic_filter_ignores_metric})

# %% [markdown]
# Вывод пробы: `RecallAt` отвергается `StochasticRank`, `LambdaMart` и
# `YetiRank`, а `StochasticFilter` выдаёт идентичную модель при любом
# `metric` — он всегда оптимизирует FilteredDCG. Поэтому 25B реализует
# LambdaMART с весами `|ΔRecall@50|` поверх `PairLogit` с явными парами.

# %% [markdown]
# ## 2. OOF-выбор objective (Kaggle v1, depth 6, l2 3)
#
# Recall@50 на held-фолде dev (по ~613 запросов), максимум по checkpoints
# 100…1000 деревьев внутри фолда — одинаково оптимистично для всех objectives.
# Значения перенесены из лога Kaggle v1 (`best_held`).

# %%
kaggle_v1 = pd.DataFrame([
    ("YetiRankPairwise", 0.8706, 0.8545, 79), ("YetiRank", 0.8673, 0.8513, 55),
    ("QuerySoftMax", 0.8722, 0.8619, 522), ("LambdaRecall50_hard", 0.8722, 0.8548, 53),
    ("LambdaRecall50_soft", 0.8738, 0.8603, 53), ("StochasticFilter_RecallAt50", 0.8401, 0.8537, 131),
], columns=["loss", "fold0_best", "fold1_best", "seconds_per_job"])
kaggle_v1["oof_dev_approx"] = (kaggle_v1.fold0_best + kaggle_v1.fold1_best) / 2  # фолды равного размера
kaggle_v1 = kaggle_v1.sort_values(["oof_dev_approx", "seconds_per_job"], ascending=[False, True], kind="stable")
display(kaggle_v1.round(5))

# %% [markdown]
# ## 3. Staged OOF и финальная модель выбранного objective

# %%
run = json.loads((STAGE / "run.json").read_text())
selection = pd.read_parquet(STAGE / "selection.parquet")
best_per_loss = pd.DataFrame(run["best_per_loss"])
jobs = pd.DataFrame(run["jobs"])
print({key: run[key] for key in ("catboost", "gpus", "lambda_on_gpu", "seconds")})
display(jobs.groupby(["loss", "status"]).size().unstack(fill_value=0))
display(best_per_loss.round(5))
# Кривая OOF по числу деревьев: выбор — максимум dev, tie-break dev-tail.
display(selection.sort_values(["loss", "trees"]).round(5))
selected_loss = run["selected"]["loss"]
print("selected objective:", run["selected"])

# %% [markdown]
# ## 4. Единственная test-проверка против attempt 4
#
# Attempt 6 (те же PU bags, `YetiRankPairwise`) на test совпадает с attempt 4
# (notebook 24: 0 побед / 0 поражений на tail), поэтому это одновременно и
# парное сравнение нового objective со старым лоссом той же системы.

# %%
splits = json.loads((INPUT / "splits.json").read_text())
test_ids, tail_ids = set(splits["test"]), set(splits["tail"])
labels = pd.read_parquet(ROOT / "artifacts/validation/labels.parquet")
relevant = labels.groupby("eval_query_id").item_id.agg(set).to_dict()

test_top = pd.read_parquet(STAGE / "test_top50.parquet")
test_by_loss = {loss: lists(frame) for loss, frame in test_top.groupby("loss")}
attempt4_test = lists(pd.read_parquet(INPUT / "attempt4_test_top50.parquet"))
# Запросы без кандидатов в пуле получают пустой список (recall 0), как в 24.
for predictions in [*test_by_loss.values(), attempt4_test]:
    for query_id in test_ids - set(predictions):
        predictions[query_id] = []


def compare(candidate: dict, baseline: dict, confidence: float) -> pd.DataFrame:
    rows = []
    for segment, ids in {"test_tail": test_ids & tail_ids, "test": test_ids}.items():
        result = paired_recall_test(candidate, baseline, relevant, query_ids=sorted(ids), k=50,
                                    confidence=confidence, n_resamples=20_000, seed=RANDOM_SEED)
        new = per_query_recall(candidate, relevant, query_ids=sorted(ids))
        old = per_query_recall(baseline, relevant, query_ids=sorted(ids))
        differences = np.asarray([new[q] - old[q] for q in sorted(ids)])
        rows.append({
            "segment": segment,
            "candidate_recall": recall_at_k(select(candidate, ids), select(relevant, ids)),
            "baseline_recall": recall_at_k(select(baseline, ids), select(relevant, ids)),
            **asdict(result),
            "wins": int((differences > 0).sum()), "losses": int((differences < 0).sum()),
        })
    return pd.DataFrame(rows)


gate = compare(test_by_loss[selected_loss], attempt4_test, confidence=1 - ALPHA)
display(gate.round(6))
lookup = gate.set_index("segment")
tail_nonnegative = bool(lookup.loc["test_tail", "mean_delta"] >= 0)
primary = bool(lookup.loc["test_tail", "mean_delta"] > 0 and lookup.loc["test_tail", "p_value_greater"] < ALPHA)
secondary = bool(lookup.loc["test", "mean_delta"] > 0 and lookup.loc["test", "p_value_greater"] < ALPHA)
accepted = bool(tail_nonnegative and (primary or secondary))
decision = {"selected_loss": selected_loss, "tail_nonnegative": tail_nonnegative,
            "primary_significant": primary, "secondary_significant": secondary, "accepted": accepted}
print(decision)

# %% [markdown]
# ## 5. Описательная абляция objectives
#
# Каждый objective в своей лучшей OOF-конфигурации против `YetiRankPairwise`
# той же PU-схемы и того же прогона. Это ответ на вопрос «что дала замена
# лосса», но не основание для выбора: p-values без поправки, 95% CI.

# %%
ablation_rows = []
if "YetiRankPairwise" in test_by_loss:
    reference = test_by_loss["YetiRankPairwise"]
    for loss, predictions in test_by_loss.items():
        for row in compare(predictions, reference, confidence=0.95).to_dict("records"):
            ablation_rows.append({"loss": loss, **row})
else:
    print("YetiRankPairwise не досчитан в stage 25B; сравнение со старым лоссом — раздел 4")
ablation = pd.DataFrame(ablation_rows)
display(ablation.round(5))

# Прямое сравнение с ранее предложенным `StochasticFilter:metric=RecallAt;top=50`:
# лучший по OOF LambdaRecall50 против него на тех же PU bags.
lambda_losses = best_per_loss[best_per_loss.loss.str.startswith("LambdaRecall50")]
lambda_best = str(lambda_losses.iloc[0].loss) if len(lambda_losses) else None
stochastic_filter = "StochasticFilter_RecallAt50"
if lambda_best and stochastic_filter in test_by_loss:
    versus_stochastic_filter = compare(test_by_loss[lambda_best], test_by_loss[stochastic_filter], confidence=0.95)
    versus_stochastic_filter.insert(0, "candidate", lambda_best)
    display(versus_stochastic_filter.round(5))
else:
    versus_stochastic_filter = pd.DataFrame()
    print("StochasticFilter or LambdaRecall50 did not finish on Kaggle")

# %% [markdown]
# ## 6. Benchmark candidate
#
# CSV выбранного objective сохраняется всегда; корневой `answer.csv` меняется
# только при прохождении gate.

# %%
benchmark_queries = pd.read_parquet(ROOT / "dataset/benchmark_queries.parquet")
benchmark_top = pd.read_parquet(STAGE / "benchmark_top50.parquet")
benchmark_lists = lists(benchmark_top[benchmark_top.loss.eq(selected_loss)])
answer = pd.DataFrame({
    "query_id": benchmark_queries.query_id.astype(str),
    "answer": [" ".join(benchmark_lists[str(q)]) for q in benchmark_queries.query_id],
})
corpus_ids = set(pd.read_parquet(ROOT / "dataset/benchmark_items.parquet", columns=["item_id"]).item_id.astype(str))
assert len(answer) == len(benchmark_queries) == answer.query_id.nunique()
for value in answer.answer:
    item_ids = value.split()
    assert len(item_ids) == 50 == len(set(item_ids))
    assert all(re.fullmatch(r"[0-9a-f]{16}", item_id) for item_id in item_ids)
    assert set(item_ids) <= corpus_ids

candidate_path = ROOT / "answer_recall50_objective_candidate.csv"
answer.to_csv(candidate_path, index=False)
candidate_hash = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
attempt6 = pd.read_csv(ROOT / "submissions/attempt_6_pu_without_lora_v1.csv", dtype=str).set_index("query_id").answer
overlap_with_attempt6 = float(np.mean([
    len(set(benchmark_lists[q]) & set(attempt6[q].split())) / 50 for q in attempt6.index]))
if accepted:
    answer.to_csv(ROOT / "answer.csv", index=False)

report = {
    "catboost_probe": {
        "version": catboost.__version__,
        "objectives": probe_table.to_dict("records"),
        "stochastic_filter_ignores_metric": stochastic_filter_ignores_metric,
    },
    "kaggle_v1_first_param_set": kaggle_v1.to_dict("records"),
    "stage25b_run": {key: run[key] for key in (
        "catboost", "gpus", "lambda_on_gpu", "learning_rate", "border_count", "max_trees",
        "checkpoint", "round_trees", "pu_budget", "pu_seeds", "losses", "params", "seconds")},
    "jobs": run["jobs"],
    "best_per_loss": best_per_loss.to_dict("records"),
    "selected": run["selected"],
    "test_vs_attempt4": gate.to_dict("records"),
    "decision": decision,
    "ablation_vs_yetirank_pairwise": ablation.to_dict("records"),
    "lambda_recall_vs_stochastic_filter": versus_stochastic_filter.to_dict("records"),
    "candidate": {"file": candidate_path.name, "sha256": candidate_hash,
                  "benchmark_overlap_with_attempt6": overlap_with_attempt6},
}
(ROOT / "reports/recall50_objective_metrics.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print("saved", candidate_path.name, candidate_hash, "overlap with attempt 6:", round(overlap_with_attempt6, 4))
print("answer.csv replaced:", accepted)
