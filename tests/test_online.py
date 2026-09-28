"""Online-path components of the final solution (dense search, PU ensemble)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from avito_retrieval.dense_search import ExactDenseIndex
from avito_retrieval.pu_selector import FEATURES, ensemble_top50


def test_dense_index_filters_collapses_and_uses_location_slice() -> None:
    # 6 passages of 4 items; item 2 has two passages, item 3 is another category.
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.0, 1.0]])
    passage_items = np.array([0, 1, 2, 2, 3, 1])
    index = ExactDenseIndex(
        embeddings, passage_items, item_ids=np.array(["a", "b", "c", "d"]),
        item_locations=np.array([10, 20, 20, 10]), item_categories=np.array([1, 1, 1, 2]),
        item_ratings=np.array([5.0, 3.0, 4.5, 5.0]), local_weight=2.0,
    )
    global_list, local_list = index.search(np.array([1.0, 0.0]), category=1, min_rating=None, location=20)
    assert global_list == ["a", "b", "c"]          # item d filtered by category, c collapsed once
    assert local_list == ["b", "c"]                # only location-20 passages
    global_list, _ = index.search(np.array([1.0, 0.0]), category=0, min_rating=4.0, location=20)
    assert global_list == ["a", "c", "d"]          # min rating drops b


def test_ensemble_averages_ranks_and_breaks_ties_by_fused() -> None:
    class Constant:
        def __init__(self, scores: list[float]) -> None:
            self.scores = np.asarray(scores)

        def predict(self, _: pd.DataFrame, thread_count: int = -1) -> np.ndarray:
            return self.scores

    frame = pd.DataFrame(0, index=range(3), columns=FEATURES)
    frame["query_key"], frame["item_id"], frame["fused"] = "q", ["x", "y", "z"], [1, 2, 3]
    # Model ranks: x=(1,3), y=(2,2), z=(3,1). Since 1/21 + 1/23 > 2/22, x and z
    # tie above y, and the tie goes to the lower fused rank (x).
    top = ensemble_top50(frame, [Constant([3, 2, 1]), Constant([1, 2, 3])])
    assert top["q"] == ["x", "z", "y"]


def test_recall_lambda_ranks_break_ties_by_fused_and_bags_keep_positives() -> None:
    from avito_retrieval.recall_lambda import pu_rows, within_query_ranks

    group = np.array([0, 0, 0, 1, 1])
    first_row = np.array([0, 3])
    fused = np.array([1, 2, 3, 1, 2])
    ranks = within_query_ranks(group, first_row, fused, np.array([0.5, 0.5, 0.9, 0.0, 0.0]))
    assert ranks.tolist() == [2, 3, 1, 1, 2]

    rows = 120
    frame = pd.DataFrame({
        "query_key": ["q"] * rows, "fused": np.arange(1, rows + 1),
        "label": [1, 0, 1] + [0] * (rows - 3),
        "bm25": np.arange(1, rows + 1), "zero": 501, "v2": 501,
    })
    bag = pu_rows(frame, seed=41)
    assert {0, 2} <= set(bag.tolist())            # every positive is kept
    assert len(bag) == 2 + 48                     # plus the unlabeled budget
    assert np.array_equal(bag, pu_rows(frame, seed=41))
