"""B5 topic detection — BERTopic config, corpus selection, and pipeline exports."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction import text as _ft
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from shared.preprocess import preprocess


DOMAIN_STOPWORDS = {
    "openai", "like", "just", "use", "really", "think", "would", "get",
    "also", "one", "even", "know", "people", "way", "make", "time", "good",
}


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BERTOPIC_NGRAM_RANGE = (1, 2)
BERTOPIC_MIN_DF = 2
BERTOPIC_MAX_DF = 0.90
DEFAULT_RANDOM_SEED = 42


def select_corpus_df(
    df: pd.DataFrame,
    *,
    n: int,
    months: int,
) -> pd.DataFrame:
    """Keep posts from the last *months* (relative to max ``created_utc``), then *n* most recent."""
    out = df.copy()
    out["created_utc"] = pd.to_datetime(out["created_utc"], utc=True, errors="coerce")
    out = out.dropna(subset=["created_utc"])
    if out.empty:
        return out

    end = out["created_utc"].max()
    start = end - pd.DateOffset(months=months)
    windowed = out[out["created_utc"] >= start]
    return (
        windowed.sort_values("created_utc", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def min_topic_size(n_docs: int) -> int:
    # //20 was too aggressive (gives 30 for 600 docs, leaving only 2 topics).
    # //50 gives ~12 for 600 docs, matching BERTopic's recommended 10–15 floor.
    return max(5, n_docs // 50)


def make_umap_model(random_state: int = DEFAULT_RANDOM_SEED) -> UMAP:
    return UMAP(n_neighbors=15, n_components=5, min_dist=0.0, random_state=random_state)


def make_bertopic_vectorizer() -> CountVectorizer:
    stop = list(_ft.ENGLISH_STOP_WORDS.union(DOMAIN_STOPWORDS))
    return CountVectorizer(
        stop_words=stop,
        ngram_range=BERTOPIC_NGRAM_RANGE,
        min_df=BERTOPIC_MIN_DF,
        max_df=BERTOPIC_MAX_DF,
    )


def fit_bertopic(
    texts: list[str],
    *,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> tuple[BERTopic, list[int], np.ndarray | None]:
    """Fit BERTopic and return ``(model, topic_ids_per_doc, probabilities_or_None)``."""
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    model = BERTopic(
        embedding_model=embedder,
        umap_model=make_umap_model(random_state),
        min_topic_size=min_topic_size(len(texts)),
        nr_topics="auto",
        vectorizer_model=make_bertopic_vectorizer(),
        verbose=False,
    )
    topics, probs = model.fit_transform(texts)
    topic_list = [int(t) for t in topics]
    return model, topic_list, probs


def _keywords_for_topic(model: BERTopic, topic_id: int, top_n_words: int) -> list[str]:
    topic_words = model.get_topic(topic_id) or []
    return [w for w, _ in topic_words][:top_n_words]


def assign_thread_topics(
    model: BERTopic,
    topics: list[int],
    probs: np.ndarray | list | None = None,
    *,
    top_n_words: int = 10,
) -> list[dict[str, Any] | None]:
    """Assign one topic per document from BERTopic hard labels.

    Topic ``-1`` (HDBSCAN outlier) maps to ``None``. Otherwise returns
    ``{topic_id, keywords, confidence}``.
    """
    results: list[dict[str, Any] | None] = []

    for i, raw in enumerate(topics):
        tid = int(raw)
        if tid == -1:
            results.append(None)
            continue

        confidence = 1.0
        if probs is not None:
            row = probs[i]
            if isinstance(row, (list, np.ndarray)):
                arr = np.asarray(row, dtype=float).ravel()
                confidence = float(arr.max()) if arr.size else 1.0
            else:
                confidence = float(row)

        results.append({
            "topic_id": tid,
            "keywords": _keywords_for_topic(model, tid, top_n_words),
            "confidence": confidence,
        })

    return results


def detect_topics(
    texts: list[str],
    max_topics: int | None = None,
    top_n_words: int = 10,
    rep_docs_per_topic: int = 3,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Detect corpus-level topics using BERTopic (exported B5 contract).

    Returns ``[{topic_id, keywords, posts}, ...]``. Per-thread labels use
    ``assign_thread_topics()`` on the same fitted model.
    """
    cleaned = preprocess(list(texts))
    if len(cleaned) < 5:
        return []

    model, _, _ = fit_bertopic(cleaned, random_state=random_state)
    results: list[dict[str, Any]] = []
    for _, row in model.get_topic_info().iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue
        keywords = _keywords_for_topic(model, tid, top_n_words)
        rep_docs = model.get_representative_docs(tid) or []
        results.append({
            "topic_id": tid,
            "keywords": keywords,
            "posts": rep_docs[:rep_docs_per_topic],
        })

    if max_topics is not None:
        results = results[:max_topics]
    return results
