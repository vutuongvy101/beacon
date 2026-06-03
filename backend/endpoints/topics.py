from __future__ import annotations

from fastapi import APIRouter

from backend.data_loader import DATA
from backend.filters import apply_filters
from backend.helpers import dominant_sentiment, sentiment_breakdown
from backend.models import FilterState

router = APIRouter()


@router.post("/topics")
def topics(req: FilterState):
    df, _ = apply_filters(req)
    filtered_ids = set(df["post_id"].tolist()) if not df.empty else set()

    topics_out = []
    for t in DATA.get("topics", []):
        pids = [p for p in (t.get("post_ids") or []) if p in filtered_ids]
        if not pids:
            continue
        sub = df[df["post_id"].isin(pids)]
        breakdown = sentiment_breakdown(sub)
        topics_out.append({
            "topic_id": t["topic_id"],
            "label": t.get("label", f"Topic {t['topic_id']}"),
            "keywords": t.get("keywords", []),
            "post_count": len(pids),
            "sentiment_breakdown": breakdown,
            "dominant_sentiment": dominant_sentiment(breakdown),
        })

    topics_out.sort(key=lambda x: -x["post_count"])
    trend_out = []
    topic_ids = {t["topic_id"] for t in topics_out}
    for tr in DATA.get("trend", []):
        if tr.get("topic_id") not in topic_ids:
            continue
        trend_out.append({
            "topic_id": tr["topic_id"],
            "label": tr.get("label", ""),
            "weekly": tr.get("weekly", []),
        })

    return {"topics": topics_out, "trend": trend_out}
