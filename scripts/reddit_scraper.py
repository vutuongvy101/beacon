"""
Reddit brand monitoring scraper — run once offline during development.

Output: data/snapshots/reddit_<brand>_<YYYYMMDD>.jsonl

Usage:
    python scripts/reddit_scraper.py
    python scripts/reddit_scraper.py --subreddits ChatGPT OpenAI --limit 500 --brand openai
    python scripts/reddit_scraper.py --min-score 200 --min-comments 30 --top-comments 5 --max-age-days 730 --limit 100
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError as _e:
    raise SystemExit(
        f"Missing dependency: {_e}\n"
        "Run:  pip install -r requirements.txt"
    ) from _e

# Make shared/ importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.config import get_data_dir  # noqa: E402



DEFAULT_SUBREDDITS  = ["OpenAI", "ChatGPT", "codex", "dalle2"]
DEFAULT_LIMIT       = 500
DEFAULT_MIN_SCORE   = 200
DEFAULT_MIN_COMMENTS = 30
DEFAULT_TOP_COMMENTS = 5
DEFAULT_MAX_AGE_DAYS = 730   # ~2 years
SORT_MODES          = ["top", "hot", "new"]
MIN_COMMENT_WORDS   = 10
SLEEP_SEC           = 2.5   # ~24 req/min — safely under Reddit's ~30 req/min limit

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/109.0"
}



def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def extract_image_url(p: dict) -> str | None:
    """Return the first image URL from a post, or None if the post has no image."""
    url = p.get("url", "")
    if url and any(url.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp")):
        return url
    images = p.get("preview", {}).get("images", [])
    if images:
        raw = images[0].get("source", {}).get("url", "")
        return raw.replace("&amp;", "&") or None
    return None



def fetch_posts(subreddit: str, sort: str, after: str | None = None) -> tuple[list, str | None]:
    """Fetch one page (up to 100) of posts. Returns (posts, next_after)."""
    params: dict = {"limit": 100, "t": "all"}
    if after:
        params["after"] = after
    url = f"https://old.reddit.com/r/{subreddit}/{sort}.json"
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if resp.status_code == 429:
            print("  Rate limited — sleeping 60 s")
            time.sleep(60)
            return [], after   # retry same page next iteration
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code} for r/{subreddit}/{sort}")
            return [], None
        data = resp.json()["data"]
        return data["children"], data.get("after")
    except Exception as exc:
        print(f"  Error fetching r/{subreddit}/{sort}: {exc}")
        return [], None


def fetch_top_comments(subreddit: str, post_id: str, top_n: int) -> list[dict]:
    """Return up to top_n highest-scored comments for a post."""
    url = f"https://old.reddit.com/r/{subreddit}/comments/{post_id}.json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 429:
            print(f"  Rate limited fetching comments for {post_id} — sleeping 60 s")
            time.sleep(60)
            resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code} fetching comments for {post_id} — skipping")
            return []
        candidates = []
        for item in resp.json()[1]["data"]["children"][:50]:
            if item["kind"] != "t1":
                continue
            body  = clean_text(item["data"].get("body", ""))
            score = item["data"].get("score", 0)
            if not body or body in ("[deleted]", "[removed]"):
                continue
            if len(body.split()) < MIN_COMMENT_WORDS:
                continue
            candidates.append({
                "comment_id": item["data"].get("id", ""),
                "body":       body,
                "score":      score,
            })
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_n]
    except Exception as exc:
        print(f"  Error fetching comments for {post_id}: {exc}")
        return []



def seen_ids_path(snapshot_dir: Path, brand: str) -> Path:
    return snapshot_dir / f"reddit_{brand}_seen_ids.txt"


def load_seen_ids(path: Path) -> set[str]:
    """Load all previously crawled post IDs from the persistent seen-IDs file."""
    if not path.exists():
        return set()
    ids = set(path.read_text(encoding="utf-8").splitlines())
    print(f"Loaded {len(ids)} seen post IDs from {path.name}")
    return ids


def record_seen_id(path: Path, post_id: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(post_id + "\n")



def load_checkpoint(path: Path) -> list[dict]:
    """Load previously saved records from today's output file (resume support)."""
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if records:
        print(f"Resumed from checkpoint: {len(records)} posts already in today's file.")
    return records


def append_record(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_by_subreddit(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in records:
        sub = rec["subreddit"]
        counts[sub] = counts.get(sub, 0) + 1
    return counts


def per_subreddit_targets(subreddits: list[str], limit: int) -> dict[str, int]:
    """Split *limit* evenly across subreddits (remainder → first subs)."""
    n = len(subreddits)
    base, rem = divmod(limit, n)
    return {
        sub: base + (1 if i < rem else 0)
        for i, sub in enumerate(subreddits)
    }


def remaining_per_subreddit(
    subreddits: list[str],
    limit: int,
    collected: dict[str, int],
) -> dict[str, int]:
    targets = per_subreddit_targets(subreddits, limit)
    return {
        sub: max(0, targets[sub] - collected.get(sub, 0))
        for sub in subreddits
    }



def crawl(
    subreddits: list[str],
    limit: int,
    output_path: Path,
    ids_path: Path,
    min_score: int,
    min_comments: int,
    top_comments: int,
    max_age_days: int,
) -> None:
    cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).timestamp()

    # Seen IDs persist across all runs; today's records resume within the same day.
    seen_ids = load_seen_ids(ids_path)
    records  = load_checkpoint(output_path)

    # Any IDs already in today's file are folded into seen_ids (handles partial reruns).
    for rec in records:
        seen_ids.add(rec["post_id"])

    targets   = per_subreddit_targets(subreddits, limit)
    remaining = remaining_per_subreddit(subreddits, limit, count_by_subreddit(records))

    print("Per-subreddit targets:")
    for sub in subreddits:
        have = targets[sub] - remaining[sub]
        print(f"  r/{sub}: {have}/{targets[sub]}")

    # Round-robin: one listing page per subreddit per pass so no single sub fills the quota first.
    # sort_i / after track pagination per subreddit independently.
    sort_i: dict[str, int] = {sub: 0 for sub in subreddits}
    after: dict[str, str | None] = {sub: None for sub in subreddits}

    while sum(remaining.values()) > 0:
        progressed = False

        for subreddit in subreddits:
            need = remaining[subreddit]
            if need <= 0:
                continue

            si = sort_i[subreddit]
            if si >= len(SORT_MODES):
                continue

            sort = SORT_MODES[si]
            posts, next_after = fetch_posts(subreddit, sort, after=after[subreddit])
            after[subreddit] = next_after

            if not posts:
                if next_after is None:
                    sort_i[subreddit] += 1
                    after[subreddit] = None
                    print(f"  r/{subreddit}: exhausted sort={sort}")
                continue

            progressed = True
            saved_this_page = 0

            for post in posts:
                if remaining[subreddit] <= 0:
                    break

                p       = post["data"]
                post_id = p.get("id", "")

                if not post_id or post_id in seen_ids:
                    continue
                if p.get("created_utc", 0) < cutoff_ts:
                    continue
                if p.get("score", 0) < min_score:
                    continue
                if p.get("num_comments", 0) < min_comments:
                    continue

                title    = clean_text(p.get("title", ""))
                selftext = clean_text(p.get("selftext", ""))

                time.sleep(1)  # brief pause before comment fetch to avoid burst rate-limiting
                record = {
                    "post_id":      post_id,
                    "title":        title,
                    "selftext":     selftext,
                    "text":         f"{title} {selftext}".strip(),
                    "subreddit":    subreddit,
                    "score":        p.get("score", 0),
                    "num_comments": p.get("num_comments", 0),
                    "created_utc":  datetime.utcfromtimestamp(
                                        p.get("created_utc", 0)
                                    ).isoformat(),
                    "url":          f"https://reddit.com{p.get('permalink', '')}",
                    "image_url":    extract_image_url(p),
                    "top_comments": fetch_top_comments(subreddit, post_id, top_comments),
                }

                append_record(output_path, record)
                record_seen_id(ids_path, post_id)
                records.append(record)
                seen_ids.add(post_id)
                remaining[subreddit] -= 1
                saved_this_page += 1

                n = len(records)
                if n % 25 == 0 or n <= 3:
                    print(
                        f"  [{n}/{limit}] r/{subreddit} "
                        f"({targets[subreddit] - remaining[subreddit]}/{targets[subreddit]})"
                    )

                time.sleep(SLEEP_SEC)

            if saved_this_page:
                print(
                    f"  r/{subreddit} sort={sort}: +{saved_this_page} "
                    f"({targets[subreddit] - remaining[subreddit]}/{targets[subreddit]})"
                )

            if next_after is None:
                sort_i[subreddit] += 1
                after[subreddit] = None

        if not progressed:
            print("\nNo more posts available from any subreddit.")
            break

    print(f"\nDone. {len(records)} posts saved → {output_path}")
    print("Final per-subreddit counts:")
    for sub in subreddits:
        have = targets[sub] - remaining[sub]
        print(f"  r/{sub}: {have}/{targets[sub]}")



def main() -> None:
    parser = argparse.ArgumentParser(description="Reddit brand monitoring scraper")
    parser.add_argument("--subreddits", nargs="+", default=DEFAULT_SUBREDDITS,
                        help="Subreddit names to crawl (default: ChatGPT OpenAI artificial)")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help="Total posts to collect (default: 500)")
    parser.add_argument("--brand", default="openai",
                        help="Brand label used in the output filename (default: openai)")
    parser.add_argument("--min-score", type=int, default=DEFAULT_MIN_SCORE,
                        help="Minimum post upvote score to include (default: 200)")
    parser.add_argument("--min-comments", type=int, default=DEFAULT_MIN_COMMENTS,
                        help="Minimum number of comments a post must have (default: 30)")
    parser.add_argument("--top-comments", type=int, default=DEFAULT_TOP_COMMENTS,
                        help="Number of top-voted comments to collect per post (default: 5)")
    parser.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS,
                        help="Only include posts newer than this many days (default: 730 / ~2 years)")
    args = parser.parse_args()

    snapshot_dir = get_data_dir() / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    output_path = snapshot_dir / f"reddit_{args.brand}_{date.today():%Y%m%d}.jsonl"
    ids_path    = seen_ids_path(snapshot_dir, args.brand)

    print(f"Output      : {output_path}")
    print(f"Seen-IDs    : {ids_path}")
    print(f"Subs        : {args.subreddits}")
    print(f"Target      : {args.limit} posts")
    print(f"Sort        : {SORT_MODES}")
    print(f"Min score   : {args.min_score}")
    print(f"Min comments: {args.min_comments}")
    print(f"Top comments: {args.top_comments}")
    print(f"Max age     : {args.max_age_days} days\n")

    crawl(
        subreddits=args.subreddits,
        limit=args.limit,
        output_path=output_path,
        ids_path=ids_path,
        min_score=args.min_score,
        min_comments=args.min_comments,
        top_comments=args.top_comments,
        max_age_days=args.max_age_days,
    )


if __name__ == "__main__":
    main()
