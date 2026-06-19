"""Section 5 baseline — B4 sentiment + B5 BERTopic + TF-IDF crisis LR comparison."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from advanced.layer3_llm.multitask_models import TOPIC_LABELS
from advanced.layer3_llm.train_utils import evaluate_predictions
from shared.preprocessing import clean_for_llm
from shared.sentiment import classify_sentiment_detailed
from shared.topics import assign_thread_topics, fit_bertopic

TOPIC_TO_ID = {t: i for i, t in enumerate(TOPIC_LABELS)}


def keywords_to_layer3(keywords: list[str]) -> str:
    """Map B5 c-TF-IDF keywords → Layer 3 topic slug."""
    blob = " ".join(keywords).lower()
    if any(k in blob for k in ["price", "subscription", "billing", "cost", "expensive"]):
        return "pricing_subscriptions"
    if any(k in blob for k in ["api", "developer", "codex", "token", "rate", "limit"]):
        return "api_developer"
    if any(k in blob for k in ["gpt", "release", "model", "feature", "sora", "dall", "chatgpt"]):
        return "product_releases"
    if any(k in blob for k in ["safety", "ethics", "alignment", "bias", "privacy", "weapon"]):
        return "safety_ethics"
    if any(k in blob for k in ["altman", "musk", "board", "ceo", "corporate", "sam"]):
        return "corporate_leadership"
    if any(k in blob for k in ["anthropic", "google", "gemini", "grok", "claude", "competition"]):
        return "competition"
    if any(k in blob for k in ["outage", "down", "error", "slow", "broken", "reliability"]):
        return "reliability_outages"
    return "general_discussion"


def b4_label_to_score(label: str, confidence: float = 0.5) -> float:
    """Map B4 pos/neg to continuous score for MAE vs gold."""
    conf = float(np.clip(confidence, 0.0, 1.0))
    if str(label).lower() == "positive":
        return conf
    if str(label).lower() == "negative":
        return -conf
    return 0.0


def train_crisis_lr(
    train_df: pd.DataFrame,
    *,
    text_col: str = "text",
    random_state: int = 42,
) -> tuple[TfidfVectorizer, LogisticRegression]:
    """TF-IDF + LR for crisis severity (basic technique — no existing B-layer module)."""
    from shared.sentiment import clean_tweet

    texts = [clean_tweet(t) for t in train_df[text_col].astype(str)]
    y = train_df["crisis_severity"].astype(int).values
    vec = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=500, random_state=random_state)
    clf.fit(X, y)
    return vec, clf


def save_crisis_lr(vec: TfidfVectorizer, clf: LogisticRegression, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vec, "model": clf}, path)


def load_crisis_lr(path: Path) -> tuple[TfidfVectorizer, LogisticRegression]:
    bundle = joblib.load(path)
    return bundle["vectorizer"], bundle["model"]


def predict_crisis_lr(
    texts: list[str],
    vec: TfidfVectorizer,
    clf: LogisticRegression,
) -> list[int]:
    from shared.sentiment import clean_tweet

    cleaned = [clean_tweet(t) for t in texts]
    return clf.predict(vec.transform(cleaned)).astype(int).tolist()


def fit_b5_model(
    train_df: pd.DataFrame,
    *,
    text_col: str = "text",
    random_state: int = 42,
) -> BERTopic:
    """Fit BERTopic on train pool (same contract as B5 / shared.topics.fit_bertopic)."""
    texts = clean_for_llm(train_df[text_col].astype(str).tolist())
    model, _, _ = fit_bertopic(texts, random_state=random_state)
    return model


def save_b5_model(model: BERTopic, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Default folder serialization (not pickle — pickle expects a file path).
    model.save(str(path))


def load_b5_model(path: Path) -> BERTopic:
    return BERTopic.load(str(path))


def b5_checkpoint_ready(path: Path) -> bool:
    """True when a prior BERTopic export exists (file or folder save)."""
    return path.exists() and (path.is_file() or path.is_dir())


def predict_b5_topic_slugs(model: BERTopic, texts: list[str]) -> list[str]:
    """Per-post topic slug via BERTopic transform + assign_thread_topics."""
    cleaned = clean_for_llm(list(texts))
    topics, probs = model.transform(cleaned)
    assignments = assign_thread_topics(model, [int(t) for t in topics], probs)
    slugs: list[str] = []
    for item in assignments:
        if item is None:
            slugs.append("general_discussion")
        else:
            slugs.append(keywords_to_layer3(item.get("keywords", [])))
    return slugs


def predict_b4_sentiment_scores(texts: list[str]) -> list[float]:
    """Run B4 LR model (basic/sentiment_model.pkl + tfidf_vectorizer.pkl)."""
    rows = classify_sentiment_detailed(texts)
    return [b4_label_to_score(r["label"], r["confidence"]) for r in rows]


def pipeline_predictions(
    eval_df: pd.DataFrame,
    *,
    vec: TfidfVectorizer,
    crisis_clf: LogisticRegression,
    bertopic_model: BERTopic,
    text_col: str = "text",
) -> dict[str, list]:
    """Per-task preds: B4 pkl + B5 BERTopic + crisis LR."""
    texts = eval_df[text_col].astype(str).tolist()

    crisis_pred = predict_crisis_lr(texts, vec, crisis_clf)
    sentiment_pred = predict_b4_sentiment_scores(texts)
    topic_slugs = predict_b5_topic_slugs(bertopic_model, texts)
    topic_pred = [TOPIC_TO_ID.get(s, TOPIC_TO_ID["general_discussion"]) for s in topic_slugs]

    return {
        "crisis_pred": crisis_pred,
        "sentiment_pred": sentiment_pred,
        "topic_pred": topic_pred,
    }


def measure_pipeline_latency_ms(
    eval_df: pd.DataFrame,
    vec: TfidfVectorizer,
    crisis_clf: LogisticRegression,
    bertopic_model: BERTopic,
    text_col: str = "text",
    n_posts: int = 50,
) -> float:
    """Mean ms/post for crisis LR + B4 pkl + B5 transform."""
    subset = eval_df.head(n_posts)
    times: list[float] = []
    for _, row in subset.iterrows():
        text = str(row[text_col])
        t0 = time.perf_counter()
        predict_crisis_lr([text], vec, crisis_clf)
        predict_b4_sentiment_scores([text])
        predict_b5_topic_slugs(bertopic_model, [text])
        times.append(time.perf_counter() - t0)
    return float(np.mean(times) * 1000)


def evaluate_pipeline_baseline(
    eval_gold: pd.DataFrame,
    train_pool: pd.DataFrame,
    *,
    crisis_ckpt: Path | None = None,
    bertopic_ckpt: Path | None = None,
    random_state: int = 42,
) -> tuple[dict[str, Any], TfidfVectorizer, LogisticRegression, BERTopic, float]:
    """Train/load B4+B5+crisis LR components; score pipeline on eval gold."""
    train_time = 0.0

    # Crisis LR (trained here on pseudo-labels — no B-layer equivalent)
    t0 = time.perf_counter()
    if crisis_ckpt and crisis_ckpt.exists():
        crisis_vec, crisis_clf = load_crisis_lr(crisis_ckpt)
    else:
        crisis_vec, crisis_clf = train_crisis_lr(train_pool, random_state=random_state)
        if crisis_ckpt:
            save_crisis_lr(crisis_vec, crisis_clf, crisis_ckpt)
        train_time += time.perf_counter() - t0

    # B5 BERTopic (fit on train pool — same shared.topics contract as B5 notebook)
    t0 = time.perf_counter()
    if bertopic_ckpt and b5_checkpoint_ready(bertopic_ckpt):
        bertopic_model = load_b5_model(bertopic_ckpt)
    else:
        bertopic_model = fit_b5_model(train_pool, random_state=random_state)
        if bertopic_ckpt:
            save_b5_model(bertopic_model, bertopic_ckpt)
        train_time += time.perf_counter() - t0

    y_topic_true = [
        TOPIC_TO_ID.get(str(t), TOPIC_TO_ID["general_discussion"])
        for t in eval_gold["topic"]
    ]
    preds = pipeline_predictions(
        eval_gold,
        vec=crisis_vec,
        crisis_clf=crisis_clf,
        bertopic_model=bertopic_model,
    )
    metrics = evaluate_predictions(
        eval_gold["crisis_severity"].tolist(),
        preds["crisis_pred"],
        eval_gold["sentiment_score"].tolist(),
        preds["sentiment_pred"],
        y_topic_true,
        preds["topic_pred"],
    )
    return metrics, crisis_vec, crisis_clf, bertopic_model, train_time


def pipeline_trainable_param_count(vec: TfidfVectorizer, clf: LogisticRegression) -> int:
    """Crisis LR coef count (B4/B5 use separate pre-trained artefacts)."""
    return int(clf.coef_.size + clf.intercept_.size)
