"""B5 topic detection — BERTopic config and ``detect_topics()`` for pipeline + notebooks."""

from __future__ import annotations

import re
from typing import Any

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction import text as _ft
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from shared.preprocess import preprocess

# ── Corpus (notebook §3.0) ───────────────────────────────────────────────────

SAMPLE_N = 400
FILTER_TO_PRODUCT = True

PRODUCT_KEYWORDS = re.compile(
    r"\b(chatgpt|gpt-?4|gpt-?4o|o1|o3|api|subscription|plus|pro\b|dall-?e|sora|"
    r"whisper|codex|plugin|custom\s*gpt|rate\s*limit|jailbreak|"
    r"model|prompt|token|billing|outage|downgrade|upgrade)\b",
    re.IGNORECASE,
)

DOMAIN_STOPWORDS = {
    "openai", "like", "just", "use", "really", "think", "would", "get",
    "also", "one", "even", "know", "people", "way", "make", "time", "good",
}

# ── Models ─────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NUM_TOPICS_LDA = 8
NUM_TOPICS_KMEANS = 8
LDA_PASSES = 15
BERTOPIC_NGRAM_RANGE = (1, 2)
BERTOPIC_MIN_DF = 2
BERTOPIC_MAX_DF = 0.90
DEFAULT_RANDOM_SEED = 42

# Heuristic labels for product-area mapping (§4.1b)
PRODUCT_AREA_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("API / developer", re.compile(r"\b(api|rate limit|token|sdk|endpoint|codex)\b", re.I)),
    ("ChatGPT / consumer", re.compile(r"\b(chatgpt|custom gpt|plus|subscription|voice|gpt)\b", re.I)),
    ("Images / multimodal", re.compile(r"\b(dall-?e|image|photo|picture|sora|video)\b", re.I)),
    ("Model behaviour", re.compile(r"\b(jailbreak|prompt|model|hallucin|reasoning)\b", re.I)),
    ("Pricing / billing", re.compile(r"\b(billing|price|subscription|plus|pro)\b", re.I)),
    ("Reliability", re.compile(r"\b(outage|down|error|limit|latency)\b", re.I)),
]


def is_product_related(text: str) -> bool:
    return bool(PRODUCT_KEYWORDS.search(text or ""))


def min_topic_size(n_docs: int) -> int:
    return max(3, n_docs // 20)


def guess_product_area(keywords: list[str]) -> str:
    blob = " ".join(keywords).lower()
    for label, pattern in PRODUCT_AREA_RULES:
        if pattern.search(blob):
            return label
    return "General product discussion"


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
) -> BERTopic:
    """Fit BERTopic with canonical §3.0 / pipeline settings."""
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    model = BERTopic(
        embedding_model=embedder,
        umap_model=make_umap_model(random_state),
        min_topic_size=min_topic_size(len(texts)),
        nr_topics="auto",
        vectorizer_model=make_bertopic_vectorizer(),
        verbose=False,
    )
    model.fit_transform(texts)
    return model


def detect_topics(
    texts: list[str],
    max_topics: int | None = None,
    top_n_words: int = 10,
    rep_docs_per_topic: int = 3,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Detect topics from social posts using BERTopic (exported B5 contract).

    Each input string should be built with ``shared.preprocess.build_doc`` and
    URL-stripped with ``shared.preprocess.preprocess`` upstream.
    """
    cleaned = preprocess(list(texts))
    if len(cleaned) < 5:
        return []

    model = fit_bertopic(cleaned, random_state=random_state)
    results: list[dict[str, Any]] = []
    for _, row in model.get_topic_info().iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue
        topic_words = model.get_topic(tid) or []
        keywords = [w for w, _ in topic_words][:top_n_words]
        rep_docs = model.get_representative_docs(tid) or []
        results.append({
            "topic_id": tid,
            "keywords": keywords,
            "posts": rep_docs[:rep_docs_per_topic],
            "product_area": guess_product_area(keywords),
        })

    if max_topics is not None:
        results = results[:max_topics]
    return results
