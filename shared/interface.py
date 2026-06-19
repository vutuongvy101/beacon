"""Phase 1 exported function contracts (see planning.md Section 7).

All basic notebooks (B1–B5) should import from this module or the underlying
``shared.*`` modules it re-exports.
"""

from __future__ import annotations

from typing import Any

from shared.preprocessing import clean_for_llm, clean_for_ner
from shared.rules import extract_brands, extract_rules, extract_signals
from shared.sentiment import (
    classify_sentiment,
    classify_sentiment_detailed,
    classify_sentiment_ft,
    predict_sentiment,
)
from shared.ner import (
    entities_as_list,
    extract_entities,
    extract_entities_batch,
    extract_entities_batch_for_posts,
    extract_entities_list,
    get_entity_summary,
)
__all__ = [
    "preprocess",
    "clean_for_ner",
    "clean_for_llm",
    "build_doc",
    "extract_entities",
    "extract_entities_list",
    "extract_entities_batch",
    "extract_entities_batch_for_posts",
    "entities_as_list",
    "get_entity_summary",
    "extract_signals",
    "extract_brands",
    "extract_rules",
    "classify_sentiment",
    "classify_sentiment_detailed",
    "predict_sentiment",
    "classify_sentiment_ft",
    "detect_topics",
    "assign_thread_topics",
    "fit_bertopic",
    "load_topic_corpus",
    "run_llm",
    "rag_retrieve",
    "cot_analyze",
    "detect_crisis",
    "analyze_multimodal",
    "process_multilingual",
]


def detect_topics(
    texts: list[str],
    max_topics: int | None = None,
    top_n_words: int = 10,
    rep_docs_per_topic: int = 3,
) -> list[dict[str, Any]]:
    from shared.topics import detect_topics as _detect_topics

    return _detect_topics(
        texts,
        max_topics=max_topics,
        top_n_words=top_n_words,
        rep_docs_per_topic=rep_docs_per_topic
    )


def assign_thread_topics(
    model: Any,
    topics: list[int],
    probs: Any = None,
    **kwargs: Any,
) -> list[dict[str, Any] | None]:
    from shared.topics import assign_thread_topics as _assign

    return _assign(model, topics, probs, **kwargs)


def fit_bertopic(texts: list[str], **kwargs: Any):
    from shared.topics import fit_bertopic as _fit

    return _fit(texts, **kwargs)


def load_topic_corpus(brand: str = "openai", **kwargs: Any):
    from shared.topics import load_topic_corpus as _load

    return _load(brand=brand, **kwargs)


def run_llm(prompt: str) -> str:
    raise NotImplementedError("A1/A2: wire Ollama client in src/ollama_client.py")


def rag_retrieve(query: str, top_k: int = 5) -> tuple[list[str], list[float]]:
    raise NotImplementedError("A2: wire FAISS retrieval in shared/rag.py")


def cot_analyze(posts: list[str]) -> dict[str, Any]:
    raise NotImplementedError("A3: wire CoT/ReAct in advanced notebooks")


def detect_crisis(posts: list[str]) -> dict[str, Any]:
    raise NotImplementedError("A4/A5: wire crisis detection in shared/crisis.py")
