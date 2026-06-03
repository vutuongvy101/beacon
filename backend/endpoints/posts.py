from __future__ import annotations

from fastapi import APIRouter

from backend.data_loader import DATA
from backend.filters import apply_filters
from backend.helpers import week_key
from backend.models import PostsRequest

router = APIRouter()


def _post_item(row) -> dict:
    pid = row["post_id"]
    ents = DATA.get("entity_by_post", {}).get(pid, [])
    text = str(row.get("clean_text", ""))
    return {
        "post_id": pid,
        "title": str(row.get("title", "")),
        "clean_text": text[:300] + ("…" if len(text) > 300 else ""),
        "score": int(row.get("score", 0) or 0),
        "num_comments": int(row.get("num_comments", 0) or 0),
        "author": str(row.get("author", "[deleted]")),
        "created_utc": int(row.get("created_utc", 0)),
        "sentiment": str(row.get("sentiment", "neutral")),
        "sentiment_confidence": float(row.get("sentiment_confidence", 0.5) or 0.5),
        "entities": ents,
        "url": str(row.get("url", "")),
    }


@router.post("/posts")
def posts(req: PostsRequest):
    df, _ = apply_filters(req)

    if req.post_id:
        df = df[df["post_id"] == req.post_id]
    if req.sentiment:
        df = df[df["sentiment"] == req.sentiment]
    if req.week:
        df = df[df["created_utc"].apply(lambda t: week_key(t) == req.week)]
    if req.entity_text:
        key = req.entity_text.lower().strip()
        pids = set(DATA.get("entity_index", {}).get(key, []))
        df = df[df["post_id"].isin(pids)]

    total = len(df)
    page = df.sort_values("score", ascending=False).iloc[req.offset : req.offset + req.limit]
    return {
        "total": total,
        "posts": [_post_item(row) for _, row in page.iterrows()],
    }
