"""B1 text preprocessing — shared by notebooks and pipeline code."""

from __future__ import annotations

import re

# HTTP(S) and www… spans; exclude chars that usually terminate a URL in prose.
_URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s\]>)\],\"']+",
    re.IGNORECASE,
)
# Markdown links: keep visible anchor text, drop the URL.
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]+\)")


def strip_urls(text: str) -> str:
    """Remove URLs from *text* and collapse whitespace."""
    if not isinstance(text, str) or not text:
        return ""
    s = _MD_LINK_RE.sub(r"\1", text)
    s = _URL_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def preprocess(texts: list[str]) -> list[str]:
    """Strip URLs / normalize whitespace for each string (B1 contract).

    Downstream topic models and sentiment code should call this (or receive
    outputs from a collector that already applied it) so URL tokens do not
    dominate the vocabulary.
    """
    return [strip_urls(t) for t in texts]
