"""
shared/preprocessing.py — Reddit text preprocessing for the Beacon system.

This module is the single source of truth for text cleaning across the project.
It exposes two preprocessing entry points (clean_for_ner, clean_for_llm) plus
helpers for signal extraction, normalization, and length filtering.

QUICK REFERENCE FOR NOTEBOOK AUTHORS
=====================================
Not sure which function to use? Find your notebook below.

    B2 (NER):
        from shared.preprocessing import clean_for_ner
        texts = clean_for_ner(raw_texts)         # preserves capitalisation for spaCy NER

    B3 (Rule-based extraction):
        from shared.preprocessing import extract_signals, clean_for_ner
        signals = [extract_signals(t) for t in raw_texts]  # hashtags, urls, mentions
        texts   = clean_for_ner(raw_texts)                 # clean prose for spaCy Matcher

    B4 (Sentiment ML):
        from shared.preprocessing import normalize, clean_for_ner
        texts = [normalize(strip_signals(clean_reddit_markdown(t))) for t in raw_texts]

    B5 (Topic clustering):
        from shared.preprocessing import clean_for_llm
        texts = clean_for_llm(raw_texts)         # readable cleaned prose for topic models

    A1 (LLM fine-tuning + prompting):
        from shared.preprocessing import clean_for_llm
        texts = clean_for_llm(raw_texts)         # readable prose

    A2 (RAG — indexing AND query time):
        from shared.preprocessing import clean_for_llm
        texts = clean_for_llm(raw_texts)         # symmetric: use for both corpus + queries

    A3 (CoT/ReAct prompting):
        from shared.preprocessing import clean_for_llm
        texts = clean_for_llm(raw_texts)         # same as A1/A2

    A5 (Crisis detection):
        from shared.preprocessing import clean_for_llm
        texts = clean_for_llm(raw_texts)         # readable prose for LLM classifier

    A6 (Pipeline orchestrator):
        # Routes through the entry points above based on the agent in question.

LOADING DATA (all notebooks):
    from shared.data_loader import load_sample
    df = load_sample(brand="openai")             # always start here

OUTPUT ARTIFACT
===============
Running this module as a script (``python shared/preprocessing.py``) regenerates
``data/snapshots/reddit_openai_clean.jsonl`` — one JSON record per line so
``wc -l`` gives an instant row count. Each record carries the original scraper
fields plus precomputed variants (``text_for_ner``, ``text_for_llm``,
``text_normalized``) and the structured ``signals`` dict.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

try:
    import contractions
    import emoji as emoji_lib
except ImportError as _e:
    raise SystemExit(
        f"Missing dependency: {_e}\n"
        "Run:  pip install -r requirements.txt"
    ) from _e


# ── pre-compiled regex patterns (≈10× faster than re.compile per call) ────────

_RE_CODE_FENCE       = re.compile(r"```.*?```", re.DOTALL)        # ```python ... ```
_RE_INLINE_CODE      = re.compile(r"`[^`]+`")                     # `like this`
_RE_MD_LINK          = re.compile(r"\[([^\]]+)\]\([^)]+\)")       # [text](url) → keep text
_RE_MD_BOLD_ITALIC   = re.compile(r"\*{1,3}([^*]+)\*{1,3}")       # **bold**, *italic*
_RE_MD_HEADER        = re.compile(r"^#{1,6}\s+", re.MULTILINE)    # # H1, ## H2
_RE_QUOTE_BLOCK      = re.compile(r"^>+\s?", re.MULTILINE)
_RE_MULTI_WHITESPACE = re.compile(r"\s+")

_RE_URL              = re.compile(r"https?://[^\s)\]]+|www\.[^\s)\]]+")
_RE_HASHTAG          = re.compile(r"#\w+")
_RE_REDDIT_USER      = re.compile(r"(?:^|\s)(/?u/\w+)", re.IGNORECASE)
_RE_REDDIT_SUB       = re.compile(r"(?:^|\s)(/?r/\w+)", re.IGNORECASE)
_RE_TWITTER_MENTION  = re.compile(r"(?<!\w)@\w+")                 # @sama (cross-posted content)


# ── Stage 1: structural cleaning ──────────────────────────────────────────────

def clean_reddit_markdown(text: str) -> str:
    """Strip Reddit-specific markdown formatting and collapse whitespace.

    Preserves the visible content (link text, bold/italic text) but removes the
    markdown syntax around it. Code fences are replaced with ``[CODE]`` rather
    than deleted so downstream techniques can still detect code-related content.

    Called by:
        Internal — used by clean_for_ner and clean_for_llm as the first pass.
    """
    if not text:
        return ""
    text = _RE_CODE_FENCE.sub(" [CODE] ", text)
    text = _RE_INLINE_CODE.sub(" ", text)
    text = _RE_MD_LINK.sub(r"\1", text)
    text = _RE_MD_BOLD_ITALIC.sub(r"\1", text)
    text = _RE_MD_HEADER.sub("", text)
    text = _RE_QUOTE_BLOCK.sub("", text)
    return _RE_MULTI_WHITESPACE.sub(" ", text).strip()


# ── Stage 2: structured signal extraction ─────────────────────────────────────

def extract_signals(text: str) -> dict:
    """Extract URLs, hashtags, Reddit users, subreddits, and @mentions.

    Pulls structured signals out of raw Reddit text into separate lists. Call
    this BEFORE any cleaning step — signals are removed during cleaning and
    won't be recoverable afterwards.

    The output is precomputed for the full corpus and saved in
    ``data/snapshots/reddit_openai_clean.jsonl`` under the ``signals`` key,
    so B3 can load it directly without recomputation.

    Called by:
        B3 (Rule extraction) — primary consumer; builds hashtag trend
            visualizations, mention networks, and URL frequency analyses.
        B1 notebook — generates reddit_openai_clean.jsonl.
        A6 (Orchestrator) — passes signals into shared graph state.

    Returns:
        dict with keys: urls, hashtags, reddit_users, reddit_subs, mentions.
        All lists are lowercased except urls (URLs are case-sensitive).
    """
    if not text:
        return {"urls": [], "hashtags": [], "reddit_users": [],
                "reddit_subs": [], "mentions": []}
    return {
        "urls":         _RE_URL.findall(text),
        "hashtags":     [h.lower() for h in _RE_HASHTAG.findall(text)],
        "reddit_users": [u.strip().lower() for u in _RE_REDDIT_USER.findall(text)],
        "reddit_subs":  [s.strip().lower() for s in _RE_REDDIT_SUB.findall(text)],
        "mentions":     [m.lower() for m in _RE_TWITTER_MENTION.findall(text)],
    }


def strip_signals(text: str) -> str:
    """Remove URLs, hashtags, and mentions from text after extracting them.

    Called by:
        Internal — used by clean_for_ner and clean_for_llm after
        extract_signals() has already captured the structured content.
    """
    if not text:
        return ""
    text = _RE_URL.sub(" ", text)
    text = _RE_HASHTAG.sub(" ", text)
    text = _RE_REDDIT_USER.sub(" ", text)
    text = _RE_REDDIT_SUB.sub(" ", text)
    text = _RE_TWITTER_MENTION.sub(" ", text)
    return _RE_MULTI_WHITESPACE.sub(" ", text).strip()


# ── Stage 3: normalization ────────────────────────────────────────────────────

def normalize(text: str, emoji_strategy: str = "to_text") -> str:
    """Lowercase, expand contractions, handle emoji.

    Contractions are expanded ("don't" → "do not") so negation markers like
    "not" remain explicit in the text for downstream sentiment analysis.

    Args:
        emoji_strategy: ``"to_text"`` (🔥 → "fire"), ``"strip"`` (remove
            entirely), or ``"keep"`` (no-op). Default is ``"to_text"`` so
            sentiment-carrying emoji survive as searchable text.

    Called by:
        Internal — available for notebooks that need fully normalised text.
        Not called by clean_for_ner (NER needs original capitalisation).
        Not called directly by clean_for_llm (which applies contraction + emoji
        handling inline without lowercasing).
    """
    if not text:
        return ""
    text = text.lower()
    text = contractions.fix(text)
    if emoji_strategy == "to_text":
        text = emoji_lib.demojize(text, delimiters=(" ", " "))
    elif emoji_strategy == "strip":
        text = emoji_lib.replace_emoji(text, "")
    return _RE_MULTI_WHITESPACE.sub(" ", text).strip()


def _normalize_text(text: str) -> str:
    """Apply markdown cleaning, signal removal, and full normalisation."""
    return normalize(strip_signals(clean_reddit_markdown(text)))


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC ENTRY POINTS — these are the functions notebooks should call.
# ─────────────────────────────────────────────────────────────────────────────

def clean_for_ner(texts: list[str]) -> list[str]:
    """Clean Reddit markdown for NER and rule-based extraction.

    Strips markdown, code fences, URLs, hashtags, and Reddit mentions,
    but PRESERVES original capitalisation and sentence structure. Does NOT
    lowercase or apply full normalisation.

    Capitalisation is critical for spaCy NER — "OpenAI" and "Sam Altman" are
    recognised as entities because of their casing. Passing lowercased text to
    en_core_web_sm reduces entity recall significantly.

    Called by:
        B2 (NER)              — feed output directly to spaCy nlp() pipeline.
        B3 (Rule extraction)  — feed output to spaCy Matcher or regex patterns.

    Example:
        >>> clean_for_ner(["**Sam Altman** said OpenAI's GPT-5 https://openai.com"])
        ["Sam Altman said OpenAI's GPT-5"]
    """
    out: list[str] = []
    for t in texts:
        if not t:
            out.append("")
            continue
        cleaned = clean_reddit_markdown(t)
        cleaned = strip_signals(cleaned)
        out.append(cleaned)
    return out


def clean_for_llm(texts: list[str]) -> list[str]:
    """Clean Reddit posts into readable prose for LLM and embedding inputs.

    Strips markdown and signals, expands contractions, converts emoji to text.
    Does NOT lowercase. Sentence structure is preserved so the text reads naturally.

    LLMs (Qwen, DistilBERT, sentence-transformers) are pretrained on natural
    prose and need readable input rather than aggressively normalised token strings.

    Called by:
        A1 (LLM fine-tuning) — training examples and inference inputs.
        A1 (LLM prompting)   — post text inserted into prompt templates.
        A2 (RAG)             — BOTH corpus indexing AND query preprocessing.
                               These must be symmetric or retrieval degrades.
        A3 (CoT/ReAct)       — posts passed to reasoning prompts.
        A5 (Crisis det.)     — LLM classifier path.

    Example:
        >>> clean_for_llm(["**Holy shit** GPT-5 is NOT worth $200/mo 🔥"])
        ["Holy shit GPT-5 is NOT worth $200/mo fire"]
    """
    out: list[str] = []
    for t in texts:
        if not t:
            out.append("")
            continue
        cleaned = clean_reddit_markdown(t)
        cleaned = strip_signals(cleaned)
        cleaned = contractions.fix(cleaned)
        cleaned = emoji_lib.demojize(cleaned, delimiters=(" ", " "))
        cleaned = _RE_MULTI_WHITESPACE.sub(" ", cleaned).strip()
        out.append(cleaned)
    return out


# ── post-cleaning utility ─────────────────────────────────────────────────────

def filter_empty(texts: list[str], min_tokens: int = 3) -> list[str]:
    """Drop posts shorter than ``min_tokens`` whitespace-split tokens.

    Reddit posts include image-only submissions and deleted content
    (``[removed]``) that produce empty or one-token strings after cleaning.
    Passing these to downstream models wastes compute and can trigger edge-case
    errors in some libraries.

    Called by:
        Any notebook after calling clean_for_*() on a corpus.
        Pipeline orchestrator after the Preprocessing Agent runs.
    """
    return [t for t in texts if t and len(t.split()) >= min_tokens]


# ─────────────────────────────────────────────────────────────────────────────
#  SCRIPT MODE — regenerate the cleaned snapshot or run a sanity demo.
#
#    python shared/preprocessing.py                # demo (one example post)
#    python shared/preprocessing.py --build        # build reddit_openai_clean.jsonl
#    python shared/preprocessing.py --build --brand openai
# ─────────────────────────────────────────────────────────────────────────────

def _demo() -> None:
    """Side-by-side reference of every entry point on the same input."""
    sample = (
        "**Sam Altman** said OpenAI's GPT-5 is NOT worth $200/mo 🔥 "
        "https://openai.com #ChatGPT u/sama"
    )
    print("=== preprocessing.py — entry-point reference ===\n")
    print(f"RAW INPUT:\n  {sample}\n")
    print(f"clean_for_ner   →  {clean_for_ner([sample])[0]}")
    print("  used by:  B2 (NER), B3 (rule-based)\n")
    print(f"clean_for_llm   →  {clean_for_llm([sample])[0]}")
    print("  used by:  A1, A2 (RAG), A3, A5\n")
    print(f"normalize       →  {_normalize_text(sample)}")
    print("  used by:  notebooks needing lowercased, expanded text\n")
    print(f"extract_signals →  {extract_signals(sample)}")
    print("  used by:  B3 (rule-based extraction)\n")


def _build_clean_snapshot(brand: str, include_comments: bool = True) -> None:
    """Regenerate data/snapshots/reddit_<brand>_clean.jsonl from raw snapshots.

    Output format is JSONL (one record per line) so the row count can be
    verified instantly with ``wc -l`` (Linux/Mac) or
    ``Get-Content ... | Measure-Object -Line`` (PowerShell) without parsing JSON.

    Fields written per record:
        All original scraper fields (post_id, title, selftext, subreddit,
        score, num_comments, created_utc, url, image_url, top_comments) plus:
        • text_raw           — title + selftext only (no comments)
        • text_with_comments — title + selftext + top comment bodies joined
                               with " | ".
        • text_for_ner       — clean_for_ner(text_with_comments)
        • text_for_llm       — clean_for_llm(text_with_comments)
        • text_normalized    — normalize after markdown cleaning + signal removal
        • signals            — extract_signals(text_raw)
        • n_comments_included — how many comment bodies were appended

    Args:
        brand:            Brand label matching the snapshot filename pattern
                          (e.g. "openai" → reddit_openai_*.jsonl).
        include_comments: If False, text_with_comments == text_raw. Useful
                          for ablation studies comparing post-only vs
                          post+comment preprocessing in the B1 notebook.
    """
    import sys
    from pathlib import Path
    _project_root = str(Path(__file__).resolve().parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    from shared.config import get_data_dir
    from shared.data_loader import load_all_reddit

    MIN_COMMENT_WORDS = 5

    posts = load_all_reddit(brand=brand, as_df=False)
    out_path = get_data_dir() / "snapshots" / f"reddit_{brand}_clean.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = n_skipped = 0
    total_comments_included = 0

    with open(out_path, "w", encoding="utf-8") as fh:
        for p in posts:
            text_raw = p.get("text", "").strip()

            if not text_raw:
                n_skipped += 1
                continue

            if include_comments:
                comment_bodies = []
                for c in p.get("top_comments", []):
                    body = c.get("body", "").strip()
                    if not body or body in ("[deleted]", "[removed]"):
                        continue
                    if len(body.split()) < MIN_COMMENT_WORDS:
                        continue
                    comment_bodies.append(body)

                text_with_comments = (
                    " | ".join([text_raw] + comment_bodies)
                    if comment_bodies
                    else text_raw
                )
                n_comments_included = len(comment_bodies)
            else:
                text_with_comments = text_raw
                n_comments_included = 0

            total_comments_included += n_comments_included
            signals = extract_signals(text_raw)

            record = {k: v for k, v in p.items() if k != "text"}
            record["text_raw"]            = text_raw
            record["text_with_comments"]  = text_with_comments
            record["n_comments_included"] = n_comments_included
            record["text_for_ner"]        = clean_for_ner([text_with_comments])[0]
            record["text_for_llm"]        = clean_for_llm([text_with_comments])[0]
            record["text_normalized"]     = _normalize_text(text_with_comments)
            record["signals"]             = signals

            fh.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
            n_written += 1

    print(f"✓ wrote {n_written} cleaned posts → {out_path}")
    if n_skipped:
        print(f"  {n_skipped} posts skipped (empty text_raw)")
    print(f"  {total_comments_included} comment bodies appended across all posts")
    avg = total_comments_included / n_written if n_written else 0
    print(f"  avg {avg:.1f} comments per post included")
    print(f"\nVerify row count:")
    print(f"  Linux/Mac : wc -l {out_path}")
    print(f"  PowerShell: Get-Content '{out_path}' | Measure-Object -Line")


def _json_default(obj):
    """Make pandas.Timestamp / datetime serialisable when as_df=False isn't honoured."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reddit text preprocessing module")
    parser.add_argument(
        "--build", action="store_true",
        help="Regenerate reddit_<brand>_clean.jsonl from raw snapshots.",
    )
    parser.add_argument(
        "--brand", default="openai",
        help="Brand label (default: openai).",
    )
    parser.add_argument(
        "--no-comments", action="store_true",
        help="Build without appending top comment bodies (post text only).",
    )
    args = parser.parse_args()

    if args.build:
        _build_clean_snapshot(args.brand, include_comments=not args.no_comments)
    else:
        _demo()
