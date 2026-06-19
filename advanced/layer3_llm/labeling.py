"""External ChatGPT labeling helpers — export batch, validate, load pseudo-labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from advanced.layer3_llm.multitask_models import TOPIC_LABELS

REQUIRED_PSEUDO_COLUMNS = [
    "post_id",
    "text",
    "crisis_severity",
    "sentiment_score",
    "topic",
    "rationale",
]

# ── Shared rubric (notebook API + external ChatGPT workflow) ─────────────────

FEW_SHOT_EXAMPLES: list[dict[str, Any]] = [
    {
        "text": "GPT-4o voice mode is amazing, best update yet.",
        "crisis_severity": 0,
        "sentiment_score": 0.9,
        "topic": "product_releases",
        "rationale": "Positive product praise, no brand concern.",
    },
    {
        "text": "Funny meme about ChatGPT — not really complaining, just joking.",
        "crisis_severity": 0,
        "sentiment_score": 0.2,
        "topic": "general_discussion",
        "rationale": "Lighthearted discussion, no operational or reputational risk.",
    },
    {
        "text": "Subscription went up again, not sure the Plus plan is worth $20.",
        "crisis_severity": 1,
        "sentiment_score": -0.4,
        "topic": "pricing_subscriptions",
        "rationale": "Individual pricing frustration — use severity 1, not 0 or 2.",
    },
    {
        "text": "Codex limits dropped after April 9 — frustrating but I'm still using it.",
        "crisis_severity": 1,
        "sentiment_score": -0.5,
        "topic": "api_developer",
        "rationale": "Personal developer gripe about limits; not a mass outage.",
    },
    {
        "text": "Sam Altman floated a silly codename for the next model — people are memeing.",
        "crisis_severity": 1,
        "sentiment_score": -0.3,
        "topic": "corporate_leadership",
        "rationale": "Mild leadership mockery; annoying but not escalating concern.",
    },
    {
        "text": "OpenAI deleted 8 models overnight with no warning — people are grieving.",
        "crisis_severity": 2,
        "sentiment_score": -0.7,
        "topic": "product_releases",
        "rationale": "Widespread user backlash over forced migration.",
    },
    {
        "text": "Trust in OpenAI keeps eroding after another safety policy flip-flop.",
        "crisis_severity": 2,
        "sentiment_score": -0.6,
        "topic": "safety_ethics",
        "rationale": "Escalating reputational concern, not yet an active emergency.",
    },
    {
        "text": "Is ChatGPT down? Can't send anything — outage for hours.",
        "crisis_severity": 3,
        "sentiment_score": -0.8,
        "topic": "reliability_outages",
        "rationale": "Active service outage affecting users now.",
    },
    {
        "text": "Head of robotics resigned — allegation OpenAI builds lethal autonomous weapons.",
        "crisis_severity": 3,
        "sentiment_score": -0.9,
        "topic": "safety_ethics",
        "rationale": "Active ethics/reputational crisis allegation.",
    },
    {
        "text": "Anthropic Claude is better than GPT for coding — switched last week.",
        "crisis_severity": 0,
        "sentiment_score": 0.3,
        "topic": "competition",
        "rationale": "Competitive comparison without OpenAI operational crisis.",
    },
]


def system_prompt() -> str:
    topic_list = ", ".join(TOPIC_LABELS)
    return (
        "You are a brand-monitoring analyst labeling Reddit posts about OpenAI / ChatGPT.\n"
        "Return STRICT JSON only — one array of objects with keys: "
        "post_id, text, crisis_severity (0-3), sentiment_score (-1 to 1), "
        f"topic (exactly one of: {topic_list}), rationale.\n\n"
        "crisis_severity calibration:\n"
        "  0 = no brand concern (praise, neutral tips, memes, competitive chatter)\n"
        "  1 = minor complaint (single-user billing/API/limit frustration) — USE OFTEN\n"
        "  2 = escalating concern (mass backlash, trust erosion, policy controversy)\n"
        "  3 = active crisis (outages, breaches, safety scandals, boycott calls)\n"
        "Do NOT collapse mild negatives into 0. Do NOT jump to 3 for individual annoyance.\n\n"
        "sentiment_score: dominant tone toward OpenAI in the thread (-1 angry, 0 mixed, +1 praise).\n\n"
        "topic rules:\n"
        "  competition = vs Anthropic/Google/Grok/open-source alternatives\n"
        "  product_releases = new models/features/Sora/DALL·E\n"
        "  api_developer = API, Codex, rate limits, tokens\n"
        "  safety_ethics = alignment, privacy, harm, weapons, censorship\n"
        "  corporate_leadership = Altman, board, governance\n"
        "  reliability_outages = downtime, errors, 'is it down?'\n"
        "  pricing_subscriptions = Plus/Pro pricing, billing\n"
        "  general_discussion = only when none of the above fit"
    )


def build_labeling_prompt(batch_records: list[dict]) -> str:
    few_shot = "\n".join(json.dumps(x, ensure_ascii=False) for x in FEW_SHOT_EXAMPLES)
    payload = json.dumps(batch_records, ensure_ascii=False)
    return (
        "Examples:\n"
        + few_shot
        + "\n\nLabel every post below. Return ONLY a JSON array in the same order:\n"
        + payload
    )


def audit_pseudo_distribution(df: pd.DataFrame) -> dict[str, Any]:
    """Flag skewed pseudo-label distributions that usually hurt training."""
    n = max(len(df), 1)
    crisis = df["crisis_severity"].value_counts().sort_index()
    topic = df["topic"].value_counts()
    issues: list[str] = []

    sev1_pct = crisis.get(1, 0) / n
    if sev1_pct < 0.05:
        issues.append(
            f"crisis severity 1 is {sev1_pct:.1%} of labels (<5%) — "
            "GPT is likely under-labeling minor complaints; consider re-labeling."
        )
    if crisis.get(0, 0) / n > 0.80:
        issues.append("crisis severity 0 exceeds 80% — check for severity collapse toward 'no concern'.")

    top_topic_share = topic.iloc[0] / n if len(topic) else 0.0
    if top_topic_share > 0.35:
        issues.append(
            f"topic '{topic.index[0]}' is {top_topic_share:.1%} of labels — "
            "topic taxonomy may be over-used as a default."
        )

    return {
        "crisis_counts": {int(k): int(v) for k, v in crisis.items()},
        "topic_top5": topic.head(5).to_dict(),
        "issues": issues,
    }


def compare_pseudo_to_gold(
    pseudo_df: pd.DataFrame,
    gold_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Merge overlapping posts and compute pseudo-vs-gold agreement metrics."""
    pseudo = pseudo_df.copy()
    gold = gold_df.copy()
    pseudo["post_id"] = pseudo["post_id"].astype(str)
    gold["post_id"] = gold["post_id"].astype(str)

    overlap = pseudo.merge(
        gold[["post_id", "crisis_severity", "sentiment_score", "topic"]],
        on="post_id",
        how="inner",
        suffixes=("_pseudo", "_gold"),
    )
    metrics: dict[str, float] = {}
    if len(overlap):
        metrics["crisis_accuracy"] = float(
            (overlap["crisis_severity_pseudo"] == overlap["crisis_severity_gold"]).mean()
        )
        metrics["topic_accuracy"] = float(
            (overlap["topic_pseudo"] == overlap["topic_gold"]).mean()
        )
        if len(overlap) > 1:
            metrics["sentiment_pearson_r"] = float(
                overlap["sentiment_score_pseudo"].corr(overlap["sentiment_score_gold"])
            )
        else:
            metrics["sentiment_pearson_r"] = float("nan")
    return overlap, metrics

def export_labeling_batch(
    df: pd.DataFrame,
    path: Path,
    *,
    text_col: str = "text_for_llm",
    n: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Write a deterministic JSON batch for external ChatGPT labeling."""
    sample = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    records = [
        {"post_id": row["post_id"], "text": str(row[text_col])[:1500]}
        for _, row in sample.iterrows()
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    return sample


def validate_pseudo_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise a pseudo-labels dataframe."""
    missing = set(REQUIRED_PSEUDO_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"pseudo_labels.csv missing columns: {sorted(missing)}")

    out = df.copy()
    out["post_id"] = out["post_id"].astype(str)
    out["crisis_severity"] = out["crisis_severity"].astype(int).clip(0, 3)
    out["sentiment_score"] = out["sentiment_score"].astype(float).clip(-1.0, 1.0)
    out["topic"] = out["topic"].astype(str)
    bad_topics = set(out["topic"]) - set(TOPIC_LABELS)
    if bad_topics:
        raise ValueError(
            f"Invalid topic values: {sorted(bad_topics)}. "
            f"Must be one of: {TOPIC_LABELS}"
        )
    if out["post_id"].duplicated().any():
        raise ValueError("Duplicate post_id values in pseudo_labels.csv")
    return out


def load_pseudo_labels(path: Path) -> pd.DataFrame:
    """Load committed pseudo-labels; fail clearly if missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}\n\n"
            "Labeling is done externally via ChatGPT (see chatgpt_labeling_prompt.md):\n"
            "  1. Export batch: posts_for_labeling.json (cell below)\n"
            "  2. Paste prompt + JSON into ChatGPT\n"
            "  3. Save response as pseudo_labels.csv in outputs/\n"
        )
    raw = path.read_text(encoding="utf-8")
    if raw.lstrip().startswith("["):
        labels = json.loads(raw)
        df = pd.DataFrame(labels)
        if "text" not in df.columns or df["text"].isna().any():
            posts_path = path.parent / "posts_for_labeling.json"
            if posts_path.exists():
                posts = json.loads(posts_path.read_text(encoding="utf-8"))
                text_map = {str(item["post_id"]): str(item.get("text", "")) for item in posts}
                df["post_id"] = df["post_id"].astype(str)
                df["text"] = df.get("text", "").map(lambda _: "") if "text" in df.columns else ""
                df["text"] = df["post_id"].map(text_map).fillna(df["text"])
    else:
        df = pd.read_csv(path)
    try:
        return validate_pseudo_labels(df)
    except Exception as exc:
        raise


def merge_chatgpt_response(
    labels_json: list[dict] | str,
    texts: dict[str, str],
    output_path: Path,
) -> pd.DataFrame:
    """Merge ChatGPT JSON response with source texts → pseudo_labels.csv."""
    if isinstance(labels_json, str):
        labels_json = json.loads(labels_json)

    rows = []
    for item in labels_json:
        pid = str(item["post_id"])
        rows.append({
            "post_id": pid,
            "text": texts.get(pid, ""),
            "crisis_severity": int(item["crisis_severity"]),
            "sentiment_score": float(item["sentiment_score"]),
            "topic": str(item["topic"]),
            "rationale": str(item.get("rationale", ""))[:300],
        })

    df = validate_pseudo_labels(pd.DataFrame(rows))
    df.to_csv(output_path, index=False)
    return df
