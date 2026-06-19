"""Pseudo-label helpers — Ollama / HuggingFace / heuristic (no OpenAI API required)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from advanced.layer3_llm.multitask_models import TOPIC_LABELS

DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
DEFAULT_HF_MODEL = os.getenv("A1_HF_LABEL_MODEL", "Qwen/Qwen2.5-3B")

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


def _few_shot_examples() -> list[dict[str, Any]]:
    if os.getenv("A1_COMPACT_RUBRIC", "0").lower() in ("1", "true", "yes"):
        return FEW_SHOT_EXAMPLES[:3]
    n = int(os.getenv("A1_FEW_SHOT", str(len(FEW_SHOT_EXAMPLES))))
    return FEW_SHOT_EXAMPLES[: max(1, min(n, len(FEW_SHOT_EXAMPLES)))]


def build_labeling_prompt(batch_records: list[dict]) -> str:
    few_shot = "\n".join(json.dumps(x, ensure_ascii=False) for x in _few_shot_examples())
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


# ── Label backends (Ollama → HF → heuristic) ───────────────────────────────────

def heuristic_label(text: str) -> dict[str, Any]:
    """Rule-based pseudo-labels — zero API cost, useful offline / on Colab."""
    t = text.lower()
    crisis_kw = ["outage", "breach", "boycott", "lawsuit", "ban", "dangerous", "hack", "leak", "down globally"]
    esc_kw = ["layoff", "backlash", "trust", "scandal", "resign", "protest", "investigation"]
    minor_kw = ["expensive", "slow", "broken", "disappoint", "cancel", "bug", "limit", "annoyed", "frustrating"]
    pos_kw = ["amazing", "love", "great", "incredible", "best", "awesome", "flawless", "insane"]

    sev = 0
    if any(k in t for k in crisis_kw) or ("down" in t and "chatgpt" in t):
        sev = 3
    elif any(k in t for k in esc_kw):
        sev = 2
    elif any(k in t for k in minor_kw):
        sev = 1

    pos = sum(1 for k in pos_kw if k in t)
    neg = sum(1 for k in minor_kw if k in t) + sum(1 for k in crisis_kw if k in t)
    sent = float(np.clip((pos - neg) / 3.0, -1, 1))

    topic = "general_discussion"
    if any(k in t for k in ["price", "subscription", "cost", "billing", "plan"]):
        topic = "pricing_subscriptions"
    elif any(k in t for k in ["api", "developer", "codex", "rate limit", "token"]):
        topic = "api_developer"
    elif any(k in t for k in ["gpt", "release", "model", "feature", "sora", "dall"]):
        topic = "product_releases"
    elif any(k in t for k in ["safety", "ethics", "alignment", "bias", "weapon", "privacy"]):
        topic = "safety_ethics"
    elif any(k in t for k in ["altman", "board", "ceo", "leadership", "corporate"]):
        topic = "corporate_leadership"
    elif any(k in t for k in ["anthropic", "google", "gemini", "grok", "claude", "deepseek"]):
        topic = "competition"
    elif any(k in t for k in ["outage", " is down", "error 500", "not working"]):
        topic = "reliability_outages"

    return {
        "crisis_severity": sev,
        "sentiment_score": sent,
        "topic": topic,
        "rationale": "heuristic rules",
    }


def parse_labels_json(content: str) -> list[dict[str, Any]]:
    """Extract a JSON array of label objects from an LLM response."""
    text = (content or "").strip()
    if not text:
        return []

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end <= start:
            return []
        parsed = json.loads(text[start : end + 1])

    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    if isinstance(parsed, dict):
        for key in ("labels", "posts", "results", "data"):
            if isinstance(parsed.get(key), list):
                return [x for x in parsed[key] if isinstance(x, dict)]
    return []


def normalize_label_row(row: dict[str, Any], text_map: dict[str, str]) -> dict[str, Any] | None:
    try:
        topic = str(row["topic"])
        if topic not in TOPIC_LABELS:
            topic = "general_discussion"
        pid = str(row["post_id"])
        return {
            "post_id": pid,
            "text": str(row.get("text") or text_map.get(pid, ""))[:1500],
            "crisis_severity": int(row["crisis_severity"]),
            "sentiment_score": float(np.clip(float(row["sentiment_score"]), -1.0, 1.0)),
            "topic": topic,
            "rationale": str(row.get("rationale", ""))[:300],
        }
    except (KeyError, TypeError, ValueError):
        return None


def resolve_label_backend() -> str:
    """Pick labeler: ollama (local) → heuristic (fast default).

    HF Qwen on CUDA is opt-in via ``A1_LABEL_BACKEND=hf`` (~45–90 min for 500 posts).
    """
    override = os.getenv("A1_LABEL_BACKEND", "auto").lower()
    if override in {"ollama", "hf", "heuristic"}:
        return override

    from shared.ollama_client import ollama_available

    if ollama_available():
        return "ollama"
    return "heuristic"


def label_backend_display_name(backend: str, *, ollama_model: str, hf_model: str) -> str:
    if backend == "ollama":
        return f"Ollama/{ollama_model}"
    if backend == "hf":
        return f"HF/{hf_model}"
    return "heuristic rules"


def _query_ollama_batch(batch_records: list[dict], *, model: str) -> list[dict[str, Any]]:
    from shared.ollama_client import query_ollama

    prompt = build_labeling_prompt(batch_records)
    content = query_ollama(prompt, system=system_prompt(), model=model, json_mode=True)
    return parse_labels_json(content)


_hf_bundle: tuple[Any, Any, Any] | None = None


def _get_hf_bundle(model_name: str):
    global _hf_bundle
    if _hf_bundle is not None:
        return _hf_bundle

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device.type == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    _hf_bundle = (tokenizer, model, device)
    return _hf_bundle


def _is_instruct_model(model_name: str) -> bool:
    return "instruct" in model_name.lower()


def _build_hf_prompt(batch_records: list[dict]) -> str:
    return (
        f"{system_prompt()}\n\n"
        f"{build_labeling_prompt(batch_records)}\n\n"
        "Respond with ONLY a JSON array of label objects. No markdown fences or commentary."
    )


def _query_hf_batch(batch_records: list[dict], *, model_name: str) -> list[dict[str, Any]]:
    import torch

    tokenizer, model, device = _get_hf_bundle(model_name)
    if _is_instruct_model(model_name):
        user_prompt = build_labeling_prompt(batch_records)
        messages = [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text = _build_hf_prompt(batch_records)

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    max_new = min(2048, 200 * len(batch_records) + 128)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
        )
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    return parse_labels_json(text)


def label_batch_records(
    batch_records: list[dict[str, str]],
    *,
    backend: str,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    hf_model: str = DEFAULT_HF_MODEL,
) -> list[dict[str, Any]]:
    """Label one batch of {post_id, text} records."""
    if backend == "heuristic":
        rows = []
        for rec in batch_records:
            h = heuristic_label(str(rec["text"]))
            rows.append({"post_id": rec["post_id"], "text": rec["text"], **h})
        return rows
    if backend == "ollama":
        return _query_ollama_batch(batch_records, model=ollama_model)
    if backend == "hf":
        return _query_hf_batch(batch_records, model_name=hf_model)
    raise ValueError(f"Unknown label backend: {backend}")


def label_dataframe(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    batch_size: int = 10,
    backend: str | None = None,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    hf_model: str = DEFAULT_HF_MODEL,
    sample_n: int | None = 500,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pseudo-label posts using Ollama, HuggingFace, or heuristic rules."""
    backend = backend or resolve_label_backend()
    work = df.sample(n=min(sample_n or len(df), len(df)), random_state=seed).reset_index(drop=True)

    records_out: list[dict[str, Any]] = []
    failures: list[str] = []

    batch_starts = range(0, len(work), batch_size)
    if backend == "heuristic":
        batch_iter = batch_starts
    else:
        from tqdm.auto import tqdm

        n_batches = (len(work) + batch_size - 1) // batch_size
        batch_iter = tqdm(
            batch_starts,
            desc=f"Pseudo-labeling ({backend})",
            total=n_batches,
        )

    for start in batch_iter:
        batch = work.iloc[start : start + batch_size]
        batch_records = [
            {"post_id": str(row["post_id"]), "text": str(row[text_col])[:1500]}
            for _, row in batch.iterrows()
        ]
        text_map = {r["post_id"]: r["text"] for r in batch_records}

        try:
            labels = label_batch_records(
                batch_records,
                backend=backend,
                ollama_model=ollama_model,
                hf_model=hf_model,
            )
        except Exception:
            if backend != "heuristic":
                labels = label_batch_records(batch_records, backend="heuristic")
            else:
                failures.extend(r["post_id"] for r in batch_records)
                continue

        if not labels:
            if backend != "heuristic":
                labels = label_batch_records(batch_records, backend="heuristic")
            else:
                failures.extend(r["post_id"] for r in batch_records)
                continue

        for item in labels:
            normalized = normalize_label_row(item, text_map)
            if normalized is None:
                failures.append(str(item.get("post_id", "unknown")))
                continue
            records_out.append(normalized)

    pseudo_df = validate_pseudo_labels(pd.DataFrame(records_out))
    meta = {
        "backend": backend,
        "label_source": label_backend_display_name(backend, ollama_model=ollama_model, hf_model=hf_model),
        "failures": failures,
        "labeled": len(records_out),
    }
    return pseudo_df, meta


def export_labeling_batch(
    df: pd.DataFrame,
    path: Path,
    *,
    text_col: str = "text_for_llm",
    n: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Write a deterministic JSON batch for optional manual / external labeling."""
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
            "Generate labels in the A1 notebook (Section 1) using Ollama, HuggingFace, or heuristic rules,\n"
            "or place a committed pseudo_labels.csv in outputs/.\n"
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
