"""B1 text preprocessing — shared by notebooks and pipeline code."""

from __future__ import annotations

import re
from typing import Any, Mapping

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


def build_doc(post: Mapping[str, Any]) -> str:
    """Assemble one document string from a Reddit-style post record.

    Expects keys ``title``, ``selftext``, and ``top_comments`` (list of dicts
    with ``body``), as produced by the Reddit scraper / ``load_sample``.

    * If ``selftext`` is longer than 20 characters, appends selftext plus the
      first two top comments (reduces topic drift toward thread reactions).
    * Otherwise (image-only / link posts), appends all available top comments
      so short titles still carry topical signal.

    Callers should run :func:`preprocess` on the returned strings before topic
    modelling or sentiment, so URLs in comment bodies are stripped.
    """
    title = str(post.get("title") or "").strip()
    parts: list[str] = [title] if title else []

    selftext = str(post.get("selftext") or "").strip()
    raw_comments = post.get("top_comments")
    comments: list[Any] = raw_comments if isinstance(raw_comments, list) else []

    if len(selftext) > 20:
        parts.append(selftext)
        for c in comments[:2]:
            if isinstance(c, Mapping):
                body = str(c.get("body") or "").strip()
                if body:
                    parts.append(body)
    else:
        for c in comments:
            if isinstance(c, Mapping):
                body = str(c.get("body") or "").strip()
                if body:
                    parts.append(body)

    return " ".join(parts)
