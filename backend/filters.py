"""Apply date range and search filters to the posts DataFrame."""

from __future__ import annotations

import pandas as pd

from backend.data_loader import DATA
from backend.models import FilterState
from backend.search import search_posts


def apply_filters(filter_state: FilterState) -> tuple[pd.DataFrame, dict]:
    posts = DATA.get("posts")
    if posts is None or posts.empty:
        return pd.DataFrame(), {
            "match_type": "none",
            "matched_entity": None,
            "matched_entity_count": None,
        }

    df = posts.copy()

    if filter_state.date_from is not None:
        df = df[df["created_utc"] >= filter_state.date_from]
    if filter_state.date_to is not None:
        df = df[df["created_utc"] <= filter_state.date_to]

    search_meta = {
        "match_type": "none",
        "matched_entity": None,
        "matched_entity_count": None,
    }

    if filter_state.search and str(filter_state.search).strip():
        result = search_posts(filter_state.search)
        ids = set(result["post_ids"])
        df = df[df["post_id"].isin(ids)]
        search_meta = {k: v for k, v in result.items() if k != "post_ids"}

    return df, search_meta
