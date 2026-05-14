"""Stubs documenting Phase 1 exported function contracts (see planning.md Section 7).

Implement the real functions inside each notebook (or pipeline modules), not here.
"""

from __future__ import annotations

from typing import Any


def preprocess(texts: list[str]) -> list[str]:
    raise NotImplementedError


def extract_entities(text: str) -> dict[str, Any]:
    raise NotImplementedError


def extract_signals(text: str) -> dict[str, Any]:
    raise NotImplementedError


def classify_sentiment(texts: list[str]) -> list[str]:
    raise NotImplementedError


def detect_topics(texts: list[str]) -> list[dict[str, Any]]:
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
