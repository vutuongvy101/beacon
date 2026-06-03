"""Load all dashboard JSON artifacts into memory once at startup."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA: dict = {}

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "outputs"
FAISS_DIR = ROOT / "data" / "faiss_index"


def _read_json(path: Path, default):
    if not path.exists():
        logger.warning("Missing data file: %s", path)
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_all() -> None:
    posts_path = OUTPUT_DIR / "cleaned_posts.json"
    posts_raw = _read_json(posts_path, [])
    posts = pd.DataFrame(posts_raw) if posts_raw else pd.DataFrame()

    sentiment_raw = _read_json(OUTPUT_DIR / "sentiment.json", [])
    if not posts.empty and sentiment_raw:
        sentiment_df = pd.DataFrame(sentiment_raw)
        posts = posts.merge(
            sentiment_df.rename(columns={"label": "sentiment", "confidence": "sentiment_confidence"}),
            on="post_id",
            how="left",
        )
        posts["sentiment"] = posts["sentiment"].fillna("neutral")
        posts["sentiment_confidence"] = posts["sentiment_confidence"].fillna(0.5)
    elif not posts.empty:
        posts["sentiment"] = "neutral"
        posts["sentiment_confidence"] = 0.5

    entities = _read_json(OUTPUT_DIR / "entities.json", [])
    entity_index: dict[str, list[str]] = {}
    for row in entities:
        pid = row.get("post_id")
        for ent in row.get("entities") or []:
            key = ent["text"].lower().strip()
            entity_index.setdefault(key, [])
            if pid and pid not in entity_index[key]:
                entity_index[key].append(pid)

    faiss_index = None
    id_map: list[str] = []
    index_path = FAISS_DIR / "posts.index"
    map_path = FAISS_DIR / "id_map.json"
    if index_path.exists() and map_path.exists():
        try:
            import faiss

            faiss_index = faiss.read_index(str(index_path))
            id_map = _read_json(map_path, [])
        except Exception as exc:
            logger.warning("FAISS load failed: %s", exc)

    if not posts.empty and "post_id" in posts.columns:
        all_ids = posts["post_id"].tolist()
    else:
        all_ids = []

    DATA.update({
        "posts": posts,
        "entities": entities,
        "entity_by_post": {r["post_id"]: r.get("entities", []) for r in entities},
        "topics": _read_json(OUTPUT_DIR / "topics.json", []),
        "trend": _read_json(OUTPUT_DIR / "trend_timeseries.json", []),
        "rag": _read_json(OUTPUT_DIR / "rag_evidence.json", []),
        "summaries": _read_json(OUTPUT_DIR / "summaries.json", []),
        "crisis": _read_json(OUTPUT_DIR / "crisis.json", {}),
        "qa_bank": _read_json(OUTPUT_DIR / "qa_bank.json", []),
        "entity_index": entity_index,
        "faiss_index": faiss_index,
        "id_map": id_map,
        "all_post_ids": all_ids,
    })
    logger.info(
        "Loaded %d posts, %d entities rows, FAISS=%s",
        len(posts),
        len(entities),
        faiss_index is not None,
    )
