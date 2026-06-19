"""External ChatGPT labeling helpers — export batch, validate, load pseudo-labels."""

from __future__ import annotations

import json
from pathlib import Path

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
