"""B4 sentiment classification — TF-IDF + Logistic Regression (Sentiment140 + domain aug)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = ROOT / "basic"

_SENTIMENT_MODEL = None
_SENTIMENT_VECTORISER = None


def clean_tweet(text: str) -> str:
    """Lightweight social-media cleaner for TF-IDF inference (B4 notebook)."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"^rt\s+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_models(model_dir: Path | str | None = None) -> tuple[Any, Any]:
    global _SENTIMENT_MODEL, _SENTIMENT_VECTORISER
    if _SENTIMENT_MODEL is not None and _SENTIMENT_VECTORISER is not None:
        return _SENTIMENT_MODEL, _SENTIMENT_VECTORISER
    base = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    model_path = base / "sentiment_model.pkl"
    vec_path = base / "tfidf_vectorizer.pkl"
    if not model_path.exists() or not vec_path.exists():
        raise FileNotFoundError(
            f"B4 models not found under {base}. "
            "Train via basic/B4_sentiment_ml.ipynb or copy .pkl files into basic/."
        )
    _SENTIMENT_MODEL = joblib.load(model_path)
    _SENTIMENT_VECTORISER = joblib.load(vec_path)
    return _SENTIMENT_MODEL, _SENTIMENT_VECTORISER


def predict_sentiment(
    texts: list[str],
    model_dir: str | Path | None = None,
) -> list[str]:
    """Predict positive/negative labels (B4 notebook export)."""
    model, vec = _load_models(model_dir)
    cleaned = [clean_tweet(t) for t in texts]
    features = vec.transform(cleaned)
    preds = model.predict(features)
    label_map = {0: "negative", 1: "positive", "0": "negative", "1": "positive"}
    return [label_map.get(p, label_map.get(int(p), str(p))) for p in preds]


def classify_sentiment(texts: list[str], model_dir: str | Path | None = None) -> list[str]:
    """Planning.md §7 contract — alias for ``predict_sentiment``."""
    return predict_sentiment(texts, model_dir=model_dir)


def classify_sentiment_detailed(
    texts: list[str],
    *,
    model_dir: str | Path | None = None,
    post_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Dashboard ``sentiment.json`` rows with confidence scores."""
    model, vec = _load_models(model_dir)
    cleaned = [clean_tweet(t) for t in texts]
    features = vec.transform(cleaned)
    preds = model.predict(features)
    try:
        proba = model.predict_proba(features)
        conf = proba.max(axis=1)
    except Exception:
        conf = [0.75] * len(preds)

    label_map = {0: "negative", 1: "positive"}
    ids = post_ids or [None] * len(texts)
    out: list[dict[str, Any]] = []
    for pid, pred, c, raw in zip(ids, preds, conf, texts):
        lab = label_map.get(pred, label_map.get(int(pred), str(pred).lower()))
        if lab not in ("positive", "negative", "neutral"):
            lab = "positive" if str(pred) in ("1", "positive") else "negative"
        row: dict[str, Any] = {
            "label": lab,
            "confidence": round(float(c), 3),
            "model_used": "lr_b4",
        }
        if pid is not None:
            row["post_id"] = pid
        out.append(row)
    return out


def classify_sentiment_ft(texts: list[str]) -> list[str]:
    """LoRA / fine-tuned model slot (A1) — not wired yet."""
    raise NotImplementedError(
        "classify_sentiment_ft requires the A1 LoRA checkpoint. "
        "Use classify_sentiment() (B4 LR) for now."
    )
