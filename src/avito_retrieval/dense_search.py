"""Exact dense passage search on CPU with the semantics of the Kaggle GPU kernels.

Kaggle stages 04/10B split the passage matrix into two shards, take the top-1500
passages of each shard, merge them by score, search the query location
separately (top-750) and collapse passages into at most 250 items under the
category and minimum-rating hard filters. This class reproduces that on CPU
for the online path (latency benchmark) and for batch re-ranking of the holdout
(notebook 27, which checks that the CPU scores do not change Recall@50).

Passages are stored sorted by item location so a location is a contiguous
slice. One matrix product serves both the global and the local channel: the
local scores are the same dot products restricted to the location slice.
"""

from __future__ import annotations

import numpy as np

TOP_PER_SHARD = 1500
SHARDS = 2
LOCAL_TOP = 750
TOP_ITEMS = 250


class ExactDenseIndex:
    def __init__(self, embeddings: np.ndarray, passage_item_rows: np.ndarray,
                 item_ids: np.ndarray, item_locations: np.ndarray, item_categories: np.ndarray,
                 item_ratings: np.ndarray, local_weight: float) -> None:
        order = np.argsort(item_locations[passage_item_rows], kind="stable")
        self.matrix = np.ascontiguousarray(embeddings[order], dtype=np.float32)
        self.passage_items = passage_item_rows[order]
        # Kaggle shards are contiguous halves of the *original* passage order.
        shard_of_original = np.zeros(len(order), dtype=np.int8)
        for shard, rows in enumerate(np.array_split(np.arange(len(order)), SHARDS)):
            shard_of_original[rows] = shard
        self.shard = shard_of_original[order]
        self.shard_rows = [np.flatnonzero(self.shard == shard) for shard in range(SHARDS)]
        locations = item_locations[self.passage_items]
        bounds = np.flatnonzero(np.diff(locations)) + 1
        starts, ends = np.r_[0, bounds], np.r_[bounds, len(locations)]
        self.slices = {int(locations[s]): slice(int(s), int(e)) for s, e in zip(starts, ends)}
        self.item_ids, self.categories, self.ratings = item_ids, item_categories, item_ratings
        self.local_weight = local_weight

    def collapse(self, passages: np.ndarray, category: int, min_rating: float | None) -> list[str]:
        answer, seen = [], set()
        for item_row in self.passage_items[passages]:
            if category and self.categories[item_row] != category:
                continue
            if min_rating is not None and self.ratings[item_row] < min_rating:
                continue
            if item_row not in seen:
                seen.add(item_row)
                answer.append(self.item_ids[item_row])
                if len(answer) == TOP_ITEMS:
                    break
        return answer

    def _global_passages(self, scores: np.ndarray) -> np.ndarray:
        candidates = []
        for rows in self.shard_rows:
            shard_scores = scores[rows]
            k = min(TOP_PER_SHARD, len(rows))
            candidates.append(rows[np.argpartition(shard_scores, -k)[-k:]])
        merged = np.concatenate(candidates)
        return merged[np.argsort(scores[merged], kind="stable")[::-1]]

    def _local_passages(self, scores: np.ndarray, location: int) -> np.ndarray:
        part = self.slices.get(location)
        if part is None:
            return np.array([], dtype=np.int64)
        local_scores = scores[part]
        k = min(LOCAL_TOP, len(local_scores))
        top = np.argpartition(local_scores, -k)[-k:]
        return top[np.argsort(local_scores[top], kind="stable")[::-1]] + part.start

    def lists_from_scores(self, scores: np.ndarray, category: int, min_rating: float | None,
                          location: int) -> tuple[list[str], list[str]]:
        """(global, local) item rankings from the scores of every passage."""
        return (self.collapse(self._global_passages(scores), category, min_rating),
                self.collapse(self._local_passages(scores, location), category, min_rating))

    def search(self, vector: np.ndarray, category: int, min_rating: float | None,
               location: int) -> tuple[list[str], list[str]]:
        return self.lists_from_scores(self.matrix @ vector.astype(np.float32), category, min_rating, location)
