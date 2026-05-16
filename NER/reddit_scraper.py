"""
Reddit scraper — COMP8420 Group Project
Brand: OpenAI | Focused on topics, products, people, events
No API credentials needed.

This scraper collects Reddit posts specifically relevant to OpenAI brand monitoring.
It avoids broad AI noise by using:
- OpenAI-focused search queries
- stricter filtering for competitor/general subreddits
- keyword density scoring
- title-priority relevance checks
- blacklist filtering for noisy benchmark/general posts

Output:
    data/snapshots/reddit_openai_raw.json
    data/snapshots/reddit_openai_3months.json

Usage:
    python reddit_scraper.py
    python reddit_scraper.py --limit 50
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError as e:
    raise SystemExit(
        f"Missing dependency: {e}\n"
        "Fix: pip install requests beautifulsoup4"
    ) from e


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH QUERIES — OpenAI scoped
# ─────────────────────────────────────────────────────────────────────────────

SEARCH_QUERIES = [
    # Products & model releases
    "OpenAI GPT-5 release",
    "OpenAI GPT-4o update",
    "OpenAI o3 benchmark",
    "OpenAI o4 mini release",
    "OpenAI Sora video generation",
    "OpenAI DALL-E 3 image generation",
    "ChatGPT new feature OpenAI",
    "OpenAI Codex agent",
    "OpenAI Whisper speech",
    "OpenAI Operator agent",
    "ChatGPT deep research OpenAI",

    # Leadership & people
    "Sam Altman OpenAI",
    "Ilya Sutskever OpenAI",
    "Greg Brockman OpenAI",
    "Mira Murati OpenAI",
    "OpenAI board leadership",
    "OpenAI leadership change",

    # Business & strategy
    "OpenAI for-profit conversion",
    "OpenAI Microsoft partnership",
    "OpenAI funding valuation",
    "OpenAI revenue profit",
    "OpenAI API pricing change",
    "OpenAI enterprise customers",

    # Competition & comparison
    "OpenAI vs Anthropic Claude",
    "OpenAI vs Google Gemini",
    "ChatGPT vs Claude OpenAI",
    "OpenAI vs Meta Llama",
    "GPT-4o vs Gemini OpenAI",
    "OpenAI Codex vs Claude Code",

    # Safety, policy & controversy
    "OpenAI safety alignment",
    "OpenAI EU regulation",
    "OpenAI lawsuit copyright",
    "Elon Musk OpenAI lawsuit",
    "OpenAI AGI artificial general intelligence",
    "OpenAI nonprofit mission",

    # Developer and technical discussions
    "OpenAI API developer",
    "OpenAI fine tuning",
    "ChatGPT hallucination OpenAI",
    "OpenAI outage downtime",
    "OpenAI RAG retrieval",
]


# Core subreddits — mostly OpenAI-related, but still lightly filtered
CORE_SUBREDDITS = {"OpenAI", "ChatGPT"}

# Strict subreddits — must strongly mention OpenAI/ChatGPT
STRICT_SUBREDDITS = {"Bard", "ClaudeAI", "LocalLLaMA", "AIAssistants"}

# General AI subreddits — medium filtering
GENERAL_SUBREDDITS = {"artificial", "singularity"}

SEARCH_SUBREDDITS = [
    "OpenAI",
    "ChatGPT",
    "artificial",
    "singularity",
    "Bard",
    "ClaudeAI",
    "LocalLLaMA",
    "AIAssistants",
]


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT & DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "snapshots"

DEFAULT_LIMIT = 1000
DEFAULT_MIN_SCORE = 20
DEFAULT_MIN_TEXT_LEN = 80
DEFAULT_MAX_AGE_DAYS = 1095
DEFAULT_POSTS_PER_QUERY = 10

MAX_TEXT_LEN = 8000

SLEEP_BETWEEN_REQUESTS = 2.0
SLEEP_ON_RATE_LIMIT = 60

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/109.0"
}


# ─────────────────────────────────────────────────────────────────────────────
# RELEVANCE FILTERS
# ─────────────────────────────────────────────────────────────────────────────

OPENAI_KEYWORDS = [
    "openai",
    "chatgpt",
    "gpt-4",
    "gpt-4o",
    "gpt-5",
    "gpt4",
    "gpt5",
    "sam altman",
    "greg brockman",
    "ilya sutskever",
    "mira murati",
    "sora",
    "dall-e",
    "dalle",
    "codex",
    "openai api",
    "whisper",
    "operator",
    "deep research",
]

COMPETITOR_CONTEXT_KEYWORDS = [
    "claude",
    "anthropic",
    "gemini",
    "google",
    "llama",
    "meta",
    "deepseek",
    "mistral",
    "qwen",
    "grok",
]

BLACKLIST_PATTERNS = [
    "i benchmarked",
    "benchmarking 31",
    "wer",
    "word error rate",
    "longmemeval",
    "recall@5",
    "leaderboard",
    "translation benchmark",
    "medical audio",
    "speech-to-text benchmark",
    "mempalace",
]


def keyword_score(text: str, keywords: list[str]) -> int:
    text = text.lower()
    return sum(text.count(k.lower()) for k in keywords)


def contains_blacklisted_noise(text: str, subreddit: str) -> bool:
    text_lower = text.lower()

    # Apply strongest noise filtering to LocalLLaMA because it often contains
    # long technical benchmark posts that only casually mention GPT/OpenAI.
    if subreddit == "LocalLLaMA":
        return any(pattern in text_lower for pattern in BLACKLIST_PATTERNS)

    # For other subreddits, only reject very obvious benchmark dump posts.
    softer_blacklist = [
        "i benchmarked",
        "word error rate",
        "medical audio",
        "translation benchmark",
    ]
    return any(pattern in text_lower for pattern in softer_blacklist)


def get_relevance_reason(title: str, selftext: str, subreddit: str) -> str | None:
    combined = f"{title} {selftext}".strip()

    title_openai_score = keyword_score(title, OPENAI_KEYWORDS)
    body_openai_score = keyword_score(selftext, OPENAI_KEYWORDS)
    total_openai_score = title_openai_score + body_openai_score

    competitor_score = keyword_score(combined, COMPETITOR_CONTEXT_KEYWORDS)

    if contains_blacklisted_noise(combined, subreddit):
        return None

    if len(combined) > MAX_TEXT_LEN:
        return None

    # Core subreddits need only a light OpenAI/product signal.
    if subreddit in CORE_SUBREDDITS:
        if total_openai_score >= 1:
            return "core subreddit with OpenAI/product keyword"
        return None

    # Strict competitor/general product communities need stronger relevance.
    if subreddit in STRICT_SUBREDDITS:
        if title_openai_score >= 1 and total_openai_score >= 2:
            return "strict subreddit with OpenAI mention in title and repeated relevance"

        if total_openai_score >= 4:
            return "strict subreddit with high OpenAI keyword density"

        # Allow direct comparison posts only when OpenAI and competitor context both exist.
        if total_openai_score >= 3 and competitor_score >= 1:
            return "strict subreddit with direct OpenAI competitor comparison"

        return None

    # General AI subreddits need moderate relevance.
    if subreddit in GENERAL_SUBREDDITS:
        if title_openai_score >= 1:
            return "general AI subreddit with OpenAI mention in title"

        if total_openai_score >= 3:
            return "general AI subreddit with repeated OpenAI relevance"

        return None

    return None


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""

    if text.strip() in ("[deleted]", "[removed]", ""):
        return ""

    text = html.unescape(text)
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_meaningful_post(title: str, selftext: str, min_len: int) -> bool:
    combined = f"{title} {selftext}".strip()

    if len(combined) < min_len:
        return False

    title_words = [w for w in title.split() if re.search(r"[a-zA-Z]", w)]
    if len(title_words) < 4:
        return False

    if not re.search(r"[a-zA-Z]", combined):
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# REDDIT SEARCH — no credentials needed
# ─────────────────────────────────────────────────────────────────────────────

def search_reddit(
    query: str,
    subreddit: str,
    limit: int = 10,
    after: str | None = None,
) -> tuple[list, str | None]:

    url = f"https://old.reddit.com/r/{subreddit}/search.json"

    params = {
        "q": query,
        "restrict_sr": "on",
        "sort": "relevance",
        "t": "all",
        "limit": min(limit, 100),
        "type": "link",
    }

    if after:
        params["after"] = after

    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=15)

        if resp.status_code == 429:
            print(f"  Rate limited — sleeping {SLEEP_ON_RATE_LIMIT}s")
            time.sleep(SLEEP_ON_RATE_LIMIT)
            return [], after

        if resp.status_code == 403:
            return [], None

        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code} for '{query}' in r/{subreddit}")
            return [], None

        data = resp.json()["data"]
        return data["children"], data.get("after")

    except requests.exceptions.Timeout:
        return [], None

    except Exception as exc:
        print(f"  Error: {exc}")
        return [], None


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT & DEDUP
# ─────────────────────────────────────────────────────────────────────────────

def load_seen_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    ids = set(path.read_text(encoding="utf-8").splitlines())
    print(f"  Loaded {len(ids)} seen IDs")
    return ids


def save_seen_id(path: Path, post_id: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(post_id + "\n")


def load_checkpoint(path: Path) -> list[dict]:
    if not path.exists():
        return []

    records = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if records:
        print(f"  Resumed: {len(records)} posts already saved")

    return records


def append_record(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CRAWL
# ─────────────────────────────────────────────────────────────────────────────

def crawl(
    limit: int,
    output_path: Path,
    ids_path: Path,
    min_score: int,
    min_text_len: int,
    max_age_days: int,
    posts_per_query: int,
) -> list[dict]:

    cutoff_ts = (
        datetime.now(timezone.utc) - timedelta(days=max_age_days)
    ).timestamp()

    three_months_ts = (
        datetime.now(timezone.utc) - timedelta(days=90)
    ).timestamp()

    seen_ids = load_seen_ids(ids_path)
    records = load_checkpoint(output_path)

    for rec in records:
        seen_ids.add(rec["id"])

    total_queries = len(SEARCH_QUERIES) * len(SEARCH_SUBREDDITS)
    queries_done = 0

    filtered_short = 0
    filtered_score = 0
    filtered_age = 0
    filtered_relevance = 0
    filtered_duplicate = 0

    print(f"\nSearch queries:  {len(SEARCH_QUERIES)}")
    print(f"Subreddits:      {len(SEARCH_SUBREDDITS)}")
    print(f"Total searches:  {total_queries}")
    print(f"Target posts:    {limit}")
    print("=" * 60)

    for query in SEARCH_QUERIES:
        if len(records) >= limit:
            break

        for subreddit in SEARCH_SUBREDDITS:
            if len(records) >= limit:
                break

            queries_done += 1
            added_this_query = 0

            posts, _ = search_reddit(
                query=query,
                subreddit=subreddit,
                limit=posts_per_query,
            )

            time.sleep(SLEEP_BETWEEN_REQUESTS)

            for post in posts:
                if len(records) >= limit:
                    break

                p = post.get("data", {})
                post_id = p.get("id", "")

                if not post_id or post_id in seen_ids:
                    filtered_duplicate += 1
                    continue

                created_ts = p.get("created_utc", 0)

                if created_ts < cutoff_ts:
                    filtered_age += 1
                    continue

                if p.get("score", 0) < min_score:
                    filtered_score += 1
                    continue

                title = clean_text(p.get("title", ""))
                selftext = clean_text(p.get("selftext", ""))

                if not is_meaningful_post(title, selftext, min_text_len):
                    filtered_short += 1
                    continue

                relevance_reason = get_relevance_reason(
                    title=title,
                    selftext=selftext,
                    subreddit=subreddit,
                )

                if not relevance_reason:
                    filtered_relevance += 1
                    continue

                created_dt = datetime.fromtimestamp(created_ts, tz=timezone.utc)

                record = {
                    "id": post_id,
                    "title": title,
                    "text": f"{title} {selftext}".strip(),
                    "selftext": selftext,
                    "subreddit": subreddit,
                    "score": p.get("score", 0),
                    "created_utc": int(created_ts),
                    "created_date": created_dt.strftime("%Y-%m-%d"),
                    "created_month": created_dt.strftime("%Y-%m"),
                    "num_comments": p.get("num_comments", 0),
                    "upvote_ratio": p.get("upvote_ratio", 0.0),
                    "url": f"https://reddit.com{p.get('permalink', '')}",
                    "search_query": query,
                    "is_recent": created_ts >= three_months_ts,
                    "relevance_reason": relevance_reason,
                }

                append_record(output_path, record)
                save_seen_id(ids_path, post_id)

                records.append(record)
                seen_ids.add(post_id)

                added_this_query += 1

            if added_this_query > 0 or queries_done % 20 == 0:
                print(
                    f"  [{len(records):>4}/{limit}] "
                    f"q{queries_done}/{total_queries} | "
                    f"'{query[:35]}' r/{subreddit} → +{added_this_query}"
                )

    print(f"\n{'=' * 60}")
    print(f"Crawl done: {len(records)} posts collected")
    print(f"  Filtered duplicates:        {filtered_duplicate}")
    print(f"  Filtered short/meme posts:  {filtered_short}")
    print(f"  Filtered low score:         {filtered_score}")
    print(f"  Filtered too old:           {filtered_age}")
    print(f"  Filtered low relevance:     {filtered_relevance}")

    return records


# ─────────────────────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(records: list[dict], output_dir: Path) -> None:
    records.sort(key=lambda x: x["created_utc"], reverse=True)

    json_path = output_dir / "reddit_openai_raw.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Full dataset:   {json_path}  ({len(records)} posts)")

    recent = [r for r in records if r.get("is_recent")]
    recent_path = output_dir / "reddit_openai_3months.json"

    with open(recent_path, "w", encoding="utf-8") as f:
        json.dump(recent, f, indent=2, ensure_ascii=False)

    print(f"✅ Recent (3 mo):  {recent_path}  ({len(recent)} posts)")

    print(f"\n{'=' * 60}")
    print("DATASET SUMMARY")
    print(f"{'=' * 60}")

    print(f"Total posts:  {len(records)}")
    print(f"Recent posts: {len(recent)}")

    print("\nBy subreddit:")
    for sub, count in Counter(r["subreddit"] for r in records).most_common():
        print(f"  r/{sub:<22} {count:>4}")

    print("\nTop search queries:")
    for query, count in Counter(r["search_query"] for r in records).most_common(10):
        print(f"  '{query:<42}' {count:>3}")

    print("\nRelevance reasons:")
    for reason, count in Counter(r["relevance_reason"] for r in records).most_common():
        print(f"  {reason:<65} {count:>4}")

    if records:
        dates = sorted(r["created_date"] for r in records)
        avg_score = sum(r["score"] for r in records) / len(records)

        print(f"\nDate range:  {dates[0]} → {dates[-1]}")
        print(f"Avg score:   {avg_score:.0f}")

    print(f"\n📁 Files saved to: {output_dir}/")
    print("   reddit_openai_raw.json      ← B2_NER.ipynb loads this")
    print("   reddit_openai_3months.json  ← trend analysis")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reddit OpenAI focused scraper — no credentials needed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reddit_scraper.py
  python reddit_scraper.py --limit 50
  python reddit_scraper.py --limit 2000 --min-score 10
        """,
    )

    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--min-score", type=int, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--min-text-len", type=int, default=DEFAULT_MIN_TEXT_LEN)
    parser.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS)
    parser.add_argument("--posts-per-query", type=int, default=DEFAULT_POSTS_PER_QUERY)

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    today = date.today().strftime("%Y%m%d")
    output_path = OUTPUT_DIR / f"reddit_openai_{today}.jsonl"
    ids_path = OUTPUT_DIR / "reddit_openai_seen_ids.txt"

    print("=" * 60)
    print("REDDIT SCRAPER — COMP8420 OpenAI Brand Monitor")
    print("No API credentials required")
    print("=" * 60)

    print(f"Output:           {output_path.name}")
    print(f"Limit:            {args.limit} posts")
    print(f"Min score:        {args.min_score}")
    print(f"Min text len:     {args.min_text_len} chars")
    print(f"Max age:          {args.max_age_days} days")
    print(f"Posts per query:  {args.posts_per_query}")

    records = crawl(
        limit=args.limit,
        output_path=output_path,
        ids_path=ids_path,
        min_score=args.min_score,
        min_text_len=args.min_text_len,
        max_age_days=args.max_age_days,
        posts_per_query=args.posts_per_query,
    )

    save_outputs(records, OUTPUT_DIR)


if __name__ == "__main__":
    main()