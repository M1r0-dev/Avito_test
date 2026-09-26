"""Filter-aware candidate channel helpers."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

RATING_RE = re.compile(r"рейтинг[^\d]{0,20}([1-5](?:[.,]\d+)?)")


def category_mask(items: pd.DataFrame, search_category: int) -> np.ndarray:
    """Apply category only when the query specifies a non-zero category."""
    if int(search_category) == 0:
        return np.ones(len(items), dtype=bool)
    return items["item_category_id"].to_numpy() == int(search_category)


def location_mask(items: pd.DataFrame, search_location_id: int) -> np.ndarray:
    """Return the local channel mask; never use this as the only channel."""
    return items["item_location_id"].to_numpy() == int(search_location_id)


def requested_min_rating(params_text: str) -> float | None:
    """Extract an explicit rating threshold from a Russian filter string."""
    match = RATING_RE.search(str(params_text).lower())
    return float(match.group(1).replace(",", ".")) if match else None

