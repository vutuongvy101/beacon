import faiss, json, time
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from shared.preprocessing import clean_for_llm
from shared.data_loader import load_all_reddit

INDEX_DIR   = Path("data/faiss_index")
INDEX_FILE  = INDEX_DIR / "corpus_minilm.index"
TEXTS_FILE  = INDEX_DIR / "corpus_texts.json"
IDS_FILE    = INDEX_DIR / "corpus_ids.json"
MODEL_NAME  = "all-MiniLM-L6-v2"

_model  = None   # loaded lazily on first call
_index  = None
_texts  = []


def _get_newest_snapshot_mtime(brand: str = "openai") -> float:
    """Return the modification time of the most recently updated snapshot file."""
    snapshot_dir = Path("data/snapshots")
    snapshots = list(snapshot_dir.glob(f"reddit_{brand}_????????.jsonl"))
    if not snapshots:
        return 0.0
    return max(p.stat().st_mtime for p in snapshots)


def _index_is_stale(brand: str = "openai") -> bool:
    """Return True if no index exists or a newer snapshot has arrived."""
    if not INDEX_FILE.exists() or not TEXTS_FILE.exists():
        return True
    index_mtime    = INDEX_FILE.stat().st_mtime
    snapshot_mtime = _get_newest_snapshot_mtime(brand)
    return snapshot_mtime > index_mtime


def _build_index(brand: str = "openai") -> None:
    """Rebuild FAISS index from all current snapshots for the given brand."""
    global _model, _index, _texts

    print(f"[RAG] Building index from latest {brand} snapshots...")
    df = load_all_reddit(brand=brand, as_df=True)

    # Build text_with_comments dynamically (no dependency on clean snapshot)
    def combine(row):
        parts = []
        if str(row.get("title", "")).strip():
            parts.append(str(row["title"]).strip())
        selftext = str(row.get("selftext", "")).strip()
        if selftext not in ("", "[removed]", "[deleted]"):
            parts.append(selftext)
        comments = row.get("top_comments", [])
        if isinstance(comments, str):
            try: comments = json.loads(comments)
            except: comments = []
        for c in (comments or [])[:5]:
            body = c.get("body", "") if isinstance(c, dict) else str(c)
            if body.strip() not in ("", "[removed]", "[deleted]"):
                parts.append(body.strip())
        return " | ".join(parts)

    df["_text"] = df.apply(combine, axis=1)
    df = df[df["_text"].str.strip() != ""].reset_index(drop=True)

    raw_texts   = df["_text"].tolist()
    clean_texts = clean_for_llm(raw_texts)
    post_ids    = df["post_id"].tolist() if "post_id" in df.columns else list(range(len(clean_texts)))

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    embeddings = _model.encode(
        clean_texts, batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    with open(TEXTS_FILE, "w", encoding="utf-8") as f:
        json.dump(clean_texts, f, ensure_ascii=False)
    with open(IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(post_ids, f)

    _index = index
    _texts = clean_texts
    print(f"[RAG] Index built: {index.ntotal} posts indexed.")


def _load_or_build(brand: str = "openai") -> None:
    """Load index from disk if fresh, rebuild if stale or missing."""
    global _model, _index, _texts

    if _index_is_stale(brand):
        _build_index(brand)
    else:
        if _model is None:
            _model = SentenceTransformer(MODEL_NAME)
        _index = faiss.read_index(str(INDEX_FILE))
        with open(TEXTS_FILE, encoding="utf-8") as f:
            _texts = json.load(f)
        print(f"[RAG] Loaded cached index: {_index.ntotal} posts.")


def rag_retrieve(query: str, top_k: int = 5, brand: str = "openai") -> list[str]:
    """
    Retrieve the top-k most relevant Reddit posts for a query.
    Automatically rebuilds the index if new snapshots are detected.

    Parameters
    ----------
    query  : str  — plain text question or crisis signal
    top_k  : int  — number of posts to return (default 5)
    brand  : str  — brand label matching snapshot filenames (default 'openai')

    Returns
    -------
    list[str] — post texts ordered by relevance, most relevant first
    """
    if _index is None:
        _load_or_build(brand)

    cleaned = clean_for_llm([query])[0]
    q_vec   = _model.encode([cleaned], normalize_embeddings=True)
    _, idxs = _index.search(q_vec, top_k)
    return [_texts[i] for i in idxs[0] if i != -1 and i < len(_texts)]