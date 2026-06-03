"""Shared aggregation helpers for panel endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def week_key(ts: int | float) -> str:
    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
    return f"{dt.isocalendar().year}-W{dt.isocalendar().week:02d}"


def sentiment_breakdown(df: pd.DataFrame) -> dict[str, int]:
    if df.empty or "sentiment" not in df.columns:
        return {"positive": 0, "negative": 0, "neutral": 0}
    counts = df["sentiment"].value_counts()
    return {
        "positive": int(counts.get("positive", 0)),
        "negative": int(counts.get("negative", 0)),
        "neutral": int(counts.get("neutral", 0)),
    }


def dominant_sentiment(counts: dict[str, int]) -> str:
    if not counts or sum(counts.values()) == 0:
        return "neutral"
    return max(counts, key=lambda k: counts.get(k, 0))


def pct(part: int, total: int) -> float:
    return round(100.0 * part / total, 1) if total else 0.0
