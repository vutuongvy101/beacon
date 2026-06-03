"""Semantic FAISS search + entity/hashtag/author exact match."""

from __future__ import annotations

from backend.data_loader import DATA

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def search_posts(query: str, top_k: int = 200) -> dict:
    if not query or not str(query).strip():
        return {
            "match_type": "none",
            "matched_entity": None,
            "matched_entity_count": None,
            "post_ids": list(DATA.get("all_post_ids", [])),
        }

    q = query.strip()
    q_lower = q.lower()
    posts = DATA["posts"]

    if q.startswith("#") and not posts.empty:
        matching = []
        for _, p in posts.iterrows():
            tags = p.get("hashtags") or []
            if isinstance(tags, str):
                tags = [tags]
            if q_lower in [str(h).lower() for h in tags]:
                matching.append(p["post_id"])
        return {
            "match_type": "hashtag",
            "matched_entity": q,
            "matched_entity_count": len(matching),
            "post_ids": matching,
        }

    if q.startswith("u/") and not posts.empty and "author" in posts.columns:
        author = q[2:].lower()
        matching = posts[posts["author"].astype(str).str.lower() == author]["post_id"].tolist()
        return {
            "match_type": "author",
            "matched_entity": q,
            "matched_entity_count": len(matching),
            "post_ids": matching,
        }

    entity_match = None
    if q_lower in DATA.get("entity_index", {}):
        entity_match = {
            "matched_entity": q,
            "matched_entity_count": len(DATA["entity_index"][q_lower]),
        }

    if DATA.get("faiss_index") is not None and DATA.get("id_map"):
        try:
            import faiss
            import numpy as np

            model = _get_embed_model()
            q_vec = model.encode([q])
            q_vec = np.asarray(q_vec, dtype=np.float32)
            faiss.normalize_L2(q_vec)
            scores, indices = DATA["faiss_index"].search(q_vec, top_k)
            id_map = DATA["id_map"]
            semantic_ids = [
                id_map[int(i)]
                for s, i in zip(scores[0], indices[0])
                if int(i) >= 0 and int(i) < len(id_map) and s > 0.3
            ]
            return {
                "match_type": "semantic",
                "matched_entity": entity_match["matched_entity"] if entity_match else None,
                "matched_entity_count": entity_match["matched_entity_count"] if entity_match else None,
                "post_ids": semantic_ids,
            }
        except Exception:
            pass

    if entity_match:
        return {
            "match_type": "entity",
            "matched_entity": entity_match["matched_entity"],
            "matched_entity_count": entity_match["matched_entity_count"],
            "post_ids": DATA["entity_index"][q_lower],
        }

    q_words = set(q_lower.split())
    fallback = []
    if not posts.empty:
        for _, p in posts.iterrows():
            text = str(p.get("clean_text", "")).lower()
            if q_lower in text or q_words & set(text.split()):
                fallback.append(p["post_id"])
    return {
        "match_type": "semantic" if fallback else "none",
        "matched_entity": entity_match["matched_entity"] if entity_match else None,
        "matched_entity_count": entity_match["matched_entity_count"] if entity_match else None,
        "post_ids": fallback or list(DATA.get("all_post_ids", [])),
    }
