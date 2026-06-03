from __future__ import annotations

from fastapi import APIRouter

from backend.data_loader import DATA
from backend.filters import apply_filters
from backend.models import EvidenceRequest
from backend.search import search_posts

router = APIRouter()


@router.post("/evidence")
def evidence(req: EvidenceRequest):
    df, _ = apply_filters(req)
    filtered_ids = set(df["post_id"].tolist()) if not df.empty else set()

    query = req.search or ""
    t = None
    if req.topic_id is not None:
        for entry in DATA.get("rag", []):
            if entry.get("topic_id") == req.topic_id:
                results = [
                    r for r in entry.get("results", [])
                    if r.get("post_id") in filtered_ids
                ]
                return {"query": entry.get("query", ""), "evidence": results}
        for topic in DATA.get("topics", []):
            if topic.get("topic_id") == req.topic_id:
                t = topic
                query = topic.get("label", "")
                break
        if t is None:
            return {"query": "", "evidence": []}

    if req.topic_id is None and req.search:
        result = search_posts(req.search, top_k=10)
        post_map = df.set_index("post_id").to_dict("index") if not df.empty else {}
        evidence_out = []
        for pid in result["post_ids"]:
            if pid not in filtered_ids or pid not in post_map:
                continue
            row = post_map[pid]
            evidence_out.append({
                "post_id": pid,
                "text": str(row.get("clean_text", ""))[:300],
                "similarity_score": 0.75,
                "sentiment": str(row.get("sentiment", "neutral")),
                "score": int(row.get("score", 0) or 0),
            })
        return {"query": query, "evidence": evidence_out[:10]}

    if req.topic_id is not None and t is not None:
        pids = [p for p in (t.get("post_ids") or []) if p in filtered_ids][:10]
        post_map = df.set_index("post_id").to_dict("index") if not df.empty else {}
        evidence_out = []
        for pid in pids:
            if pid not in post_map:
                continue
            row = post_map[pid]
            evidence_out.append({
                "post_id": pid,
                "text": str(row.get("clean_text", ""))[:300],
                "similarity_score": 0.8,
                "sentiment": str(row.get("sentiment", "neutral")),
                "score": int(row.get("score", 0) or 0),
            })
        return {"query": query, "evidence": evidence_out}

    return {"query": query or "", "evidence": []}
