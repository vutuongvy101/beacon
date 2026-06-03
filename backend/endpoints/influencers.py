from __future__ import annotations

import math
from collections import defaultdict

from fastapi import APIRouter

from backend.data_loader import DATA
from backend.filters import apply_filters
from backend.helpers import dominant_sentiment
from backend.models import FilterState

router = APIRouter()


@router.post("/influencers")
def influencers(req: FilterState):
    df, _ = apply_filters(req)
    if df.empty:
        return {"influencers": []}

    filtered_ids = set(df["post_id"].tolist())
    score_map = df.set_index("post_id")["score"].to_dict()
    sent_map = df.set_index("post_id")["sentiment"].to_dict()

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
                    "entity": text,
                    "entity_label": ent.get("label", "MISC"),
                    "mention_count": 0,
                    "upvotes": 0,
                    "sentiments": [],
                    "topic_labels": set(),
                }
            agg[key]["mention_count"] += 1
            agg[key]["upvotes"] += int(score_map.get(pid, 0) or 0)
            if pid in sent_map:
                agg[key]["sentiments"].append(sent_map[pid])

    pid_to_topics: dict[str, list[str]] = defaultdict(list)
    for t in DATA.get("topics", []):
        label = t.get("label", "")
        for pid in t.get("post_ids") or []:
            if pid in filtered_ids:
                pid_to_topics[pid].append(label)

    for row in DATA.get("entities", []):
        pid = row.get("post_id")
        if pid not in filtered_ids:
            continue
        for ent in row.get("entities") or []:
            key = ent["text"].strip().lower()
            if key in agg:
                for tl in pid_to_topics.get(pid, [])[:2]:
                    agg[key]["topic_labels"].add(tl)

    ranked = []
    for item in agg.values():
        mc = item["mention_count"]
        engagement = mc * math.log(1 + item["upvotes"])
        counts = defaultdict(int)
        for s in item["sentiments"]:
            counts[s] += 1
        breakdown = dict(counts)
        dom = dominant_sentiment({
            "positive": breakdown.get("positive", 0),
            "negative": breakdown.get("negative", 0),
            "neutral": breakdown.get("neutral", 0),
        })
        ranked.append({
            "entity": item["entity"],
            "entity_label": item["entity_label"],
            "mention_count": mc,
            "engagement_score": round(engagement, 1),
            "dominant_sentiment": dom,
            "top_topics": list(item["topic_labels"])[:3],
        })

    ranked.sort(key=lambda x: -x["engagement_score"])
    influencers = [
        {**r, "rank": i + 1}
        for i, r in enumerate(ranked[:30])
    ]
    return {"influencers": influencers}
