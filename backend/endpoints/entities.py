from __future__ import annotations

from collections import defaultdict

from fastapi import APIRouter

from backend.data_loader import DATA
from backend.filters import apply_filters
from backend.helpers import dominant_sentiment, sentiment_breakdown
from backend.models import FilterState

router = APIRouter()


@router.post("/entities")
def entities(req: FilterState):
    df, _ = apply_filters(req)
    if df.empty:
        return {"entities": []}

    filtered_ids = set(df["post_id"].tolist())
    sent_map = df.set_index("post_id")["sentiment"].to_dict() if "sentiment" in df.columns else {}

    agg: dict[str, dict] = {}
    for row in DATA.get("entities", []):
        pid = row.get("post_id")
        if pid not in filtered_ids:
            continue
        for ent in row.get("entities") or []:
            text = ent["text"].strip()
            key = text.lower()
            if key not in agg:
                agg[key] = {
                    "text": text,
                    "label": ent.get("label", "MISC"),
                    "mention_count": 0,
                    "post_ids": set(),
                    "sentiments": [],
                }
            agg[key]["mention_count"] += 1
            agg[key]["post_ids"].add(pid)
            if pid in sent_map:
                agg[key]["sentiments"].append(sent_map[pid])

    entities_out = []
    for item in agg.values():
        post_count = len(item["post_ids"])
        counts = defaultdict(int)
        for s in item["sentiments"]:
            counts[s] += 1
        breakdown = {
            "positive": counts.get("positive", 0),
            "negative": counts.get("negative", 0),
            "neutral": counts.get("neutral", 0),
        }
        entities_out.append({
            "text": item["text"],
            "label": item["label"],
            "mention_count": item["mention_count"],
            "post_count": post_count,
            "sentiment_breakdown": breakdown,
            "dominant_sentiment": dominant_sentiment(breakdown),
        })

    entities_out.sort(key=lambda x: (-x["mention_count"], -x["post_count"]))
    return {"entities": entities_out[:30]}
