"""Stubs documenting Phase 1 exported function contracts (see planning.md Section 7).

Implement the real functions inside each notebook (or pipeline modules), not here.

B1 ``preprocess`` and ``build_doc`` are implemented in ``shared.preprocess`` and
re-exported here so callers can import from either module.
"""

from __future__ import annotations

from typing import Any

from shared.preprocess import build_doc, preprocess


def extract_entities(text: str) -> dict[str, Any]:
    raise NotImplementedError


def extract_signals(text: str) -> dict[str, Any]:
    raise NotImplementedError


def classify_sentiment(texts: list[str]) -> list[str]:
    raise NotImplementedError


def detect_topics(
    texts: list[str],
    max_topics: int | None = None,
    top_n_words: int = 10,
    rep_docs_per_topic: int = 3,
) -> list[dict[str, Any]]:
    raise NotImplementedError


def classify_sentiment_ft(texts: list[str]) -> list[str]:
    raise NotImplementedError


def run_llm(prompt: str) -> str:
    raise NotImplementedError


def rag_retrieve(query: str, top_k: int = 5) -> list[str]:
    raise NotImplementedError


def cot_analyze(posts: list[str]) -> dict[str, Any]:
    raise NotImplementedError


def detect_crisis(posts: list[str]) -> dict[str, Any]:
    raise NotImplementedError


def analyze_multimodal(text: str, image_url: str) -> dict[str, Any]:
    raise NotImplementedError


def process_multilingual(text: str) -> dict[str, Any]:  # A4 stretch
    raise NotImplementedError
