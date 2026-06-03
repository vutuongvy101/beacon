from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.data_loader import DATA
from backend.models import QARequest

router = APIRouter()


@router.get("/suggested-qa")
def suggested_qa():
    questions = [
        {
            "id": q["id"],
            "text": q["text"],
            "category": q.get("category", "general"),
        }
        for q in DATA.get("qa_bank", [])
    ]
    return {"questions": questions}


@router.post("/qa")
def qa(req: QARequest):
    for q in DATA.get("qa_bank", []):
        if q.get("id") == req.question_id:
            return {
                "question": q["text"],
                "answer": q.get("answer", ""),
                "supporting_posts": q.get("supporting_posts", []),
                "suggested_search": q.get("suggested_search", ""),
            }
    raise HTTPException(status_code=404, detail="Question not found")


@router.get("/summary")
def summary():
    summaries = DATA.get("summaries", [])
    if not summaries:
        return {"summary": None}
    best = max(summaries, key=lambda s: float(s.get("judge_score", 0) or 0))
    return {
        "strategy": best.get("strategy", ""),
        "summary_text": best.get("summary_text", ""),
        "recommendation": best.get("recommendation", ""),
        "judge_score": best.get("judge_score"),
    }
