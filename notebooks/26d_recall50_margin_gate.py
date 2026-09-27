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
# # 26D. LambdaRecall50 с запасом на train: второй статистический gate
#
# Notebook 25C отклонил `LambdaRecall50_soft` (test `+0.0053`, незначимо).
# Диагностика после этого показала механизм недообучения: веса считаются по
# in-sample рангам, и уже через 100 деревьев **все** обучающие positives пула
# находятся в top-50 (in-sample Recall@50 = потолок пула 0.9265), после чего
# лосс толкает только уже правильно упорядоченные пары, а held recall стоит.
#
# Гипотеза stage 26: на train нужен запас — требовать top-`cutoff` с
# `cutoff < 50`, чтобы на новых запросах positive оставался в top-50.
# Поиск (только OOF dev, 2-fold как в notebooks 19/24/25):
#
# - 26A: cutoff `{50, 20, 10}` при τ=8 и мягкий τ=20 при cutoff 50
#   (depth 6, l2 3, до 500 деревьев с шагом 50);
# - 26B: для лучшего (cutoff 10) — `(depth, l2) ∈ {(6,10), (8,3), (8,10)}`;
# - 26C: cutoff 5, потому что OOF dev рос монотонно k50 < k20 < k10.
#
# Выбор: максимум OOF dev, tie-break OOF dev-tail и меньше деревьев. Test
# этой гипотезой уже использован один раз (25C, `alpha=.025` на endpoint),
# поэтому здесь **`alpha=.0125` на endpoint** (Bonferroni по двум просмотрам).
# `answer.csv` меняется только при неотрицательном test-tail и значимом
# primary (test-tail) или secondary (test) endpoint.

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

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from avito_retrieval.metrics import recall_at_k
from avito_retrieval.statistics import paired_recall_test, per_query_recall

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
INPUT = ROOT / "artifacts/recall50_kaggle_input"
STAGES = {
    "25B": ROOT / "artifacts/recall50_stage25b",
    "26A": ROOT / "artifacts/recall50_stage26_losses",
    "26B": ROOT / "artifacts/recall50_stage26_params",
    "26C": ROOT / "artifacts/recall50_stage26_k5",
}
RANDOM_SEED = 42
ALPHA = 0.0125


def select(mapping: dict, ids: set[str]) -> dict:
    return {query_id: mapping[query_id] for query_id in ids}


def lists(frame: pd.DataFrame) -> dict[str, list[str]]:
    ordered = frame.sort_values(["query_id", "rank"], kind="stable")
    return ordered.groupby("query_id", sort=False).item_id.agg(list).to_dict()

# %% [markdown]
# ## 1. Все OOF-оценки stage 25B/26

# %%
selection = pd.concat([
    pd.read_parquet(path / "selection.parquet").assign(stage=stage)
    for stage, path in STAGES.items()
], ignore_index=True)
selection = selection.sort_values(["dev", "dev_tail", "trees"], ascending=[False, False, True],
                                  kind="stable").reset_index(drop=True)
best = selection.groupby(["loss", "depth", "l2_leaf_reg"], sort=False).head(1).reset_index(drop=True)
display(best.round(5))
selected = best.iloc[0].to_dict()
print("selected on OOF dev:", selected)
# Для сравнения: attempt 6 (PU + YetiRankPairwise, notebook 24) OOF dev 0.86256, dev-tail 0.84659.

# %% [markdown]
# ## 2. Единственная test-проверка выбранной конфигурации против attempt 4
#
# Attempt 6 на test совпадает с attempt 4 (notebook 24), поэтому это и
# сравнение с текущим финальным selector. Финальная модель выбранной
# конфигурации обучена на всём dev тем же прогоном stage, что и OOF.

# %%
splits = json.loads((INPUT / "splits.json").read_text())
test_ids, tail_ids = set(splits["test"]), set(splits["tail"])
labels = pd.read_parquet(ROOT / "artifacts/validation/labels.parquet")
relevant = labels.groupby("eval_query_id").item_id.agg(set).to_dict()

stage_dir = STAGES[selected["stage"]]
stage_run = json.loads((stage_dir / "run.json").read_text())
final_config = next(row for row in stage_run["best_per_loss"] if row["loss"] == selected["loss"])
assert (final_config["depth"], final_config["l2_leaf_reg"], final_config["trees"]) == (
    selected["depth"], selected["l2_leaf_reg"], selected["trees"]), "final model must be the selected config"
test_top = pd.read_parquet(stage_dir / "test_top50.parquet")
candidate = lists(test_top[test_top.loss.eq(selected["loss"])])
attempt4 = lists(pd.read_parquet(INPUT / "attempt4_test_top50.parquet"))
first_look = lists(pd.read_parquet(STAGES["25B"] / "test_top50.parquet"))
for predictions in (candidate, attempt4, first_look):
    for query_id in test_ids - set(predictions):
        predictions[query_id] = []


def compare(new_lists: dict, old_lists: dict, confidence: float) -> pd.DataFrame:
    rows = []
    for segment, ids in {"test_tail": test_ids & tail_ids, "test": test_ids}.items():
        result = paired_recall_test(new_lists, old_lists, relevant, query_ids=sorted(ids), k=50,
                                    confidence=confidence, n_resamples=20_000, seed=RANDOM_SEED)
        new = per_query_recall(new_lists, relevant, query_ids=sorted(ids))
        old = per_query_recall(old_lists, relevant, query_ids=sorted(ids))
        differences = np.asarray([new[q] - old[q] for q in sorted(ids)])
        rows.append({
            "segment": segment,
            "candidate_recall": recall_at_k(select(new_lists, ids), select(relevant, ids)),
            "baseline_recall": recall_at_k(select(old_lists, ids), select(relevant, ids)),
            **asdict(result),
            "wins": int((differences > 0).sum()), "losses": int((differences < 0).sum()),
        })
    return pd.DataFrame(rows)


gate = compare(candidate, attempt4, confidence=1 - ALPHA)
display(gate.round(6))
lookup = gate.set_index("segment")
tail_nonnegative = bool(lookup.loc["test_tail", "mean_delta"] >= 0)
primary = bool(lookup.loc["test_tail", "mean_delta"] > 0 and lookup.loc["test_tail", "p_value_greater"] < ALPHA)
secondary = bool(lookup.loc["test", "mean_delta"] > 0 and lookup.loc["test", "p_value_greater"] < ALPHA)
accepted = bool(tail_nonnegative and (primary or secondary))
decision = {"alpha_per_endpoint": ALPHA, "tail_nonnegative": tail_nonnegative,
            "primary_significant": primary, "secondary_significant": secondary, "accepted": accepted}
print(decision)

# %% [markdown]
# Описательно (без решения): выбранный вариант против кандидата первого
# просмотра 25C — эффект именно запаса на train.

# %%
versus_first_look = compare(candidate, first_look, confidence=0.95)
display(versus_first_look.round(5))

# %% [markdown]
# ## 3. Benchmark candidate

# %%
benchmark_queries = pd.read_parquet(ROOT / "dataset/benchmark_queries.parquet")
benchmark_top = pd.read_parquet(stage_dir / "benchmark_top50.parquet")
benchmark_lists = lists(benchmark_top[benchmark_top.loss.eq(selected["loss"])])
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

candidate_path = ROOT / "answer_recall50_margin_candidate.csv"
answer.to_csv(candidate_path, index=False)
candidate_hash = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
attempt6 = pd.read_csv(ROOT / "submissions/attempt_6_pu_without_lora_v1.csv", dtype=str).set_index("query_id").answer
overlap_with_attempt6 = float(np.mean([
    len(set(benchmark_lists[q]) & set(attempt6[q].split())) / 50 for q in attempt6.index]))
if accepted:
    answer.to_csv(ROOT / "answer.csv", index=False)

report = {
    "hypothesis": "training cutoff < 50 keeps LambdaRecall50 gradients useful after in-sample saturation",
    "diagnosis": {"insample_recall_after_100_trees": 0.9265, "pool_ceiling": 0.9265,
                  "held_recall_plateau": [0.8657, 0.8690]},
    "oof_best_per_config": best.to_dict("records"),
    "selected": selected,
    "test_vs_attempt4": gate.to_dict("records"),
    "decision": decision,
    "vs_first_look_candidate": versus_first_look.to_dict("records"),
    "candidate": {"file": candidate_path.name, "sha256": candidate_hash,
                  "benchmark_overlap_with_attempt6": overlap_with_attempt6},
}
(ROOT / "reports/recall50_margin_metrics.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print("saved", candidate_path.name, candidate_hash, "overlap with attempt 6:", round(overlap_with_attempt6, 4))
print("answer.csv replaced:", accepted)
