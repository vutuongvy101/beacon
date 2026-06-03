from __future__ import annotations

from collections import defaultdict

from fastapi import APIRouter

from backend.data_loader import DATA
from backend.filters import apply_filters
from backend.helpers import dominant_sentiment, pct, sentiment_breakdown, week_key
from backend.models import FilterState

router = APIRouter()


@router.post("/overview")
def overview(req: FilterState):
    df, search_meta = apply_filters(req)
    crisis = DATA.get("crisis") or {}
    total = len(df)

    if total == 0:
        return {
            "kpis": {
                "total_posts": 0,
                "positive": 0,
                "negative": 0,
                "neutral": 0,
                "positive_pct": 0,
                "negative_pct": 0,
                "neutral_pct": 0,
                "unique_authors": 0,
                "total_entities": 0,
            },
            "crisis": {
                "level": crisis.get("level", "green"),
                "reason": crisis.get("reason", "No posts match current filters."),
                "negative_ratio": 0,
            },
            "sentiment_donut": [],
            "sentiment_trend": [],
            "search_meta": search_meta,
        }

    breakdown = sentiment_breakdown(df)
    pos, neg, neu = breakdown["positive"], breakdown["negative"], breakdown["neutral"]
    authors = df["author"].nunique() if "author" in df.columns else 0

    filtered_ids = set(df["post_id"].tolist())
    entity_count = 0
    seen_ent = set()
    for row in DATA.get("entities", []):
        if row.get("post_id") not in filtered_ids:
            continue
        for ent in row.get("entities") or []:
            k = ent["text"].lower()
            if k not in seen_ent:
                seen_ent.add(k)
                entity_count += 1

    weekly: dict[str, dict[str, int]] = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0})
    for _, row in df.iterrows():
        wk = week_key(row["created_utc"])
        lab = row.get("sentiment", "neutral")
        if lab in weekly[wk]:
            weekly[wk][lab] += 1

    trend = [{"week": w, **weekly[w]} for w in sorted(weekly.keys())]
    neg_ratio = neg / total if total else 0

    return {
        "kpis": {
            "total_posts": total,
            "positive": pos,
            "negative": neg,
            "neutral": neu,
            "positive_pct": pct(pos, total),
            "negative_pct": pct(neg, total),
            "neutral_pct": pct(neu, total),
            "unique_authors": int(authors),
            "total_entities": entity_count,
        },
        "crisis": {
            "level": crisis.get("level", "green"),
            "reason": crisis.get("reason", ""),
            "negative_ratio": round(neg_ratio, 3),
        },
        "sentiment_donut": [
            {"label": "positive", "count": pos, "color": "#22C55E"},
            {"label": "negative", "count": neg, "color": "#EF4444"},
            {"label": "neutral", "count": neu, "color": "#94A3B8"},
        ],
        "sentiment_trend": trend,
        "search_meta": search_meta,
    }
