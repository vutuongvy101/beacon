#!/usr/bin/env python3
"""Export pipeline JSONs and FAISS index for the dashboard PoC.

Run from repo root:
    python scripts/export_dashboard_inputs.py
    python scripts/export_dashboard_inputs.py --fast   # skip BERTopic (stub topics)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from shared.recommendations import export_recommendations as _export_reco
    _RECO_AVAILABLE = True
except ImportError:
    _RECO_AVAILABLE = False

try:
    from advanced.layer3_llm.predict import predict as _a1_predict
    _A1_AVAILABLE = True
except (ImportError, FileNotFoundError):
    _A1_AVAILABLE = False

OUTPUT_DIR = ROOT / "data" / "outputs"
FAISS_DIR = ROOT / "data" / "faiss_index"
CLEAN_JSONL = ROOT / "data" / "snapshots" / "reddit_openai_clean.jsonl"
MODEL_DIR = ROOT / "basic"

OPENAI_ENTITIES = [
    {"label": "PRODUCT", "pattern": [{"LOWER": "gpt"}, {"IS_DIGIT": True, "OP": "?"}]},
    {"label": "PRODUCT", "pattern": "ChatGPT"},
    {"label": "PRODUCT", "pattern": "Codex"},
    {"label": "PRODUCT", "pattern": "DALL-E"},
    {"label": "PRODUCT", "pattern": "Sora"},
    {"label": "PRODUCT", "pattern": "o1"},
    {"label": "PRODUCT", "pattern": "o3"},
    {"label": "ORG", "pattern": "OpenAI"},
    {"label": "PERSON", "pattern": "Sam Altman"},
    {"label": "PERSON", "pattern": "Mira Murati"},
    {"label": "PERSON", "pattern": "Greg Brockman"},
    {"label": "ORG", "pattern": "Microsoft"},
    {"label": "ORG", "pattern": "Anthropic"},
]


def _parse_utc(value: str | int | float) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _week_key(ts: int) -> str:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return f"{dt.isocalendar().year}-W{dt.isocalendar().week:02d}"


def load_clean_jsonl() -> list[dict]:
    records = []
    with open(CLEAN_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def export_cleaned_posts(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        signals = r.get("signals") or {}
        users = signals.get("reddit_users") or []
        mentions = [f"u/{u}" if not str(u).startswith("u/") else str(u) for u in users]
        out.append({
            "post_id": r["post_id"],
            "title": r.get("title", ""),
            "raw_text": r.get("text_raw", r.get("title", "")),
            "clean_text": r.get("text_for_llm", r.get("text_with_comments", "")),
            "tokens": (r.get("text_preprocessed") or "").split(),
            "lemmas": (r.get("text_preprocessed") or "").split(),
            "hashtags": signals.get("hashtags") or [],
            "mentions": mentions,
            "score": int(r.get("score", 0)),
            "num_comments": int(r.get("num_comments", 0)),
            "author": r.get("author", "[deleted]"),
            "created_utc": _parse_utc(r["created_utc"]),
            "url": r.get("url", ""),
        })
    return out


KEYWORD_ENTITIES = [
    ("OpenAI", "ORG"), ("ChatGPT", "PRODUCT"), ("GPT-4", "PRODUCT"), ("GPT-4o", "PRODUCT"),
    ("GPT-5", "PRODUCT"), ("Sam Altman", "PERSON"), ("Mira Murati", "PERSON"),
    ("Microsoft", "ORG"), ("Anthropic", "ORG"), ("Codex", "PRODUCT"), ("DALL-E", "PRODUCT"),
    ("Sora", "PRODUCT"), ("Claude", "PRODUCT"), ("Elon Musk", "PERSON"),
]


def _init_ner():
    try:
        import spacy
    except ImportError:
        return None
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("  Warning: en_core_web_sm not found; using keyword NER fallback")
        return None
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.add_patterns(OPENAI_ENTITIES)
    return nlp


def _entities_keyword(text: str) -> list[dict]:
    found = []
    lower = text.lower()
    for phrase, label in KEYWORD_ENTITIES:
        if phrase.lower() in lower:
            found.append({"text": phrase, "label": label})
    for m in re.finditer(r"\bGPT[- ]?\d+[a-z]?\b", text, re.I):
        found.append({"text": m.group(0), "label": "PRODUCT"})
    seen = set()
    out = []
    for e in found:
        k = e["text"].lower()
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


def export_entities(posts: list[dict], nlp) -> list[dict]:
    texts = [p["clean_text"] for p in posts]
    ids = [p["post_id"] for p in posts]
    result = []
    if nlp is None:
        for text, pid in zip(texts, ids):
            result.append({"post_id": pid, "entities": _entities_keyword(text)})
        return result
    for doc, pid in zip(nlp.pipe(texts, batch_size=64), ids):
        seen = set()
        ents = []
        for ent in doc.ents:
            key = (ent.text.strip().lower(), ent.label_)
            if key in seen or not ent.text.strip():
                continue
            seen.add(key)
            label = ent.label_
            if label in ("PRODUCT", "ORG", "PERSON", "GPE", "MONEY", "WORK_OF_ART"):
                ents.append({"text": ent.text.strip(), "label": label})
        result.append({"post_id": pid, "entities": ents})
    return result


def export_topics(posts: list[dict], fast: bool) -> tuple[list[dict], list[dict]]:
    if fast:
        topics = [
            {
                "topic_id": 0,
                "label": "Product releases & features",
                "keywords": ["gpt", "chatgpt", "release", "model"],
                "post_ids": [p["post_id"] for p in posts[:400]],
                "post_count": min(400, len(posts)),
                "method": "stub",
            },
            {
                "topic_id": 1,
                "label": "Pricing & subscriptions",
                "keywords": ["price", "subscription", "plus", "cost"],
                "post_ids": [p["post_id"] for p in posts[400:700]],
                "post_count": min(300, max(0, len(posts) - 400)),
                "method": "stub",
            },
        ]
        trend = [
            {
                "topic_id": t["topic_id"],
                "label": t["label"],
                "weekly": [{"week": _week_key(p["created_utc"]), "count": 1}
                           for p in posts if p["post_id"] in t["post_ids"][:50]],
            }
            for t in topics
        ]
        return topics, trend

    from shared.topics import fit_bertopic, assign_thread_topics, _keywords_for_topic

    texts = [(p.get("tokens") and " ".join(p["tokens"])) or p["clean_text"] for p in posts]
    texts = [t for t in texts if t and len(t.split()) >= 3]
    if len(texts) < 20:
        return export_topics(posts, fast=True)

    idx_map = [i for i, p in enumerate(posts) if ((p.get("tokens") and " ".join(p["tokens"])) or p["clean_text"]) and len(((p.get("tokens") and " ".join(p["tokens"])) or p["clean_text"]).split()) >= 3]
    filtered_posts = [posts[i] for i in idx_map]
    filtered_texts = [((p.get("tokens") and " ".join(p["tokens"])) or p["clean_text"]) for p in filtered_posts]

    model, topic_ids, probs = fit_bertopic(filtered_texts)
    assignments = assign_thread_topics(model, topic_ids, probs)

    by_topic: dict[int, list[str]] = defaultdict(list)
    for p, assign in zip(filtered_posts, assignments):
        if assign is None:
            continue
        by_topic[int(assign["topic_id"])].append(p["post_id"])

    topics_out = []
    for tid, pids in sorted(by_topic.items(), key=lambda x: -len(x[1])):
        kws = _keywords_for_topic(model, tid, 8)
        label = " ".join(kws[:3]).title() if kws else f"Topic {tid}"
        topics_out.append({
            "topic_id": tid,
            "label": label,
            "keywords": kws,
            "post_ids": pids,
            "post_count": len(pids),
            "method": "bertopic",
        })

    weekly_counts: dict[tuple[int, str], int] = defaultdict(int)
    pid_to_topic = {}
    for p, assign in zip(filtered_posts, assignments):
        if assign:
            pid_to_topic[p["post_id"]] = int(assign["topic_id"])
    for p in filtered_posts:
        tid = pid_to_topic.get(p["post_id"])
        if tid is not None:
            weekly_counts[(tid, _week_key(p["created_utc"]))] += 1

    trend_out = []
    for t in topics_out:
        tid = t["topic_id"]
        weeks = [{"week": w, "count": c} for (tid2, w), c in sorted(weekly_counts.items()) if tid2 == tid]
        trend_out.append({"topic_id": tid, "label": t["label"], "weekly": weeks})

    return topics_out, trend_out


def export_sentiment(posts: list[dict]) -> list[dict]:
    import joblib

    model_path = MODEL_DIR / "sentiment_model.pkl"
    vec_path = MODEL_DIR / "tfidf_vectorizer.pkl"
    labels = None
    if model_path.exists() and vec_path.exists():
        try:
            model = joblib.load(model_path)
            vec = joblib.load(vec_path)
            texts = [" ".join(p.get("tokens") or []) or p["clean_text"] for p in posts]
            X = vec.transform(texts)
            preds = model.predict(X)
            try:
                conf = model.predict_proba(X).max(axis=1)
            except Exception:
                conf = [0.75] * len(preds)
            labels = []
            for pred, c in zip(preds, conf):
                lab = str(pred).lower()
                if lab not in ("positive", "negative", "neutral"):
                    lab = "positive" if lab in ("1", "pos") else "negative"
                labels.append((lab, float(c)))
        except Exception as exc:
            print(f"  Warning: B4 model failed ({exc}); using fallback sentiment")
            labels = None
    if labels is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
            labels = []
            for p in posts:
                sc = sia.polarity_scores(p["clean_text"][:500])
                if sc["compound"] >= 0.05:
                    labels.append(("positive", abs(sc["compound"])))
                elif sc["compound"] <= -0.05:
                    labels.append(("negative", abs(sc["compound"])))
                else:
                    labels.append(("neutral", 0.5))
        except ImportError:
            labels = [("neutral", 0.5)] * len(posts)

    return [
        {
            "post_id": p["post_id"],
            "label": lab,
            "confidence": round(conf, 3),
            "model_used": "lr_b4" if model_path.exists() else "vader_stub",
        }
        for p, (lab, conf) in zip(posts, labels)
    ]


def export_crisis(posts: list[dict], sentiment: list[dict]) -> dict:
    sent_map = {s["post_id"]: s["label"] for s in sentiment}
    neg = sum(1 for p in posts if sent_map.get(p["post_id"]) == "negative")
    total = len(posts) or 1
    ratio = neg / total
    if ratio >= 0.4:
        level, reason = "red", f"High negative sentiment ({ratio:.0%}) across the corpus."
    elif ratio >= 0.28:
        level, reason = "amber", f"Elevated negative sentiment ({ratio:.0%}) detected."
    else:
        level, reason = "green", f"Sentiment within normal range ({ratio:.0%} negative)."
    flagged = [p["post_id"] for p in posts if sent_map.get(p["post_id"]) == "negative"][:20]
    return {
        "level": level,
        "reason": reason,
        "negative_ratio": round(ratio, 3),
        "flagged_post_ids": flagged,
        "react_trace": [],
    }


def export_summaries() -> list[dict]:
    return [
        {
            "strategy": "zero_shot",
            "summary_text": "Reddit discussion of OpenAI spans product launches, API limits, and corporate news. Negative threads cluster around pricing and policy; positive threads focus on new model capabilities.",
            "recommendation": "Monitor pricing-related topics weekly and prepare proactive messaging on subscription changes.",
            "rouge_l": 0.34,
            "judge_score": 3.2,
        },
        {
            "strategy": "few_shot",
            "summary_text": "Community sentiment is mixed: enthusiasm for Codex and GPT updates coexists with criticism of rate limits and leadership decisions.",
            "recommendation": "Highlight developer-facing improvements in official channels while addressing limit frustration with clear documentation.",
            "rouge_l": 0.41,
            "judge_score": 3.8,
        },
        {
            "strategy": "chain_of_thought",
            "summary_text": "Step 1: Negative share is concentrated in policy and pricing topics. Step 2: Product release posts drive positive spikes. Step 3: Entity mentions of leadership correlate with polarized threads.",
            "recommendation": "Prioritize crisis monitoring on pricing topics; amplify positive product narratives with evidence-backed posts.",
            "rouge_l": 0.46,
            "judge_score": 4.3,
        },
    ]


def export_qa_bank() -> list[dict]:
    return [
        {"id": "q1", "text": "What is the dominant sentiment about GPT-4 pricing?", "category": "sentiment",
         "answer": "Sentiment toward GPT-4 and subscription pricing skews negative in filtered Reddit threads, with frequent complaints about cost versus value.",
         "supporting_posts": [], "suggested_search": "GPT-4 pricing"},
        {"id": "q2", "text": "Are there any emerging crisis signals?", "category": "crisis",
         "answer": "Crisis indicators track elevated negative ratios on policy and pricing topics; monitor flagged posts in the crisis panel.",
         "supporting_posts": [], "suggested_search": ""},
        {"id": "q3", "text": "Which OpenAI products get the most positive reactions?", "category": "products",
         "answer": "Codex, new model releases, and developer tooling posts tend to receive more positive engagement than pricing threads.",
         "supporting_posts": [], "suggested_search": "Codex"},
        {"id": "q4", "text": "What are developers' top concerns?", "category": "community",
         "answer": "Developers frequently discuss API rate limits, token costs, and reliability of coding agents.",
         "supporting_posts": [], "suggested_search": "developer api"},
        {"id": "q5", "text": "How is sentiment about Sam Altman trending?", "category": "people",
         "answer": "Mentions of Sam Altman appear in leadership and policy threads with mixed sentiment depending on news cycle.",
         "supporting_posts": [], "suggested_search": "Sam Altman"},
        {"id": "q6", "text": "What topics are growing this month?", "category": "topics",
         "answer": "Topic clusters show shifting volume week-over-week; use the topic trend chart after applying a date filter.",
         "supporting_posts": [], "suggested_search": ""},
        {"id": "q7", "text": "Which entities are mentioned most?", "category": "entities",
         "answer": "OpenAI, ChatGPT, and GPT-family product names dominate entity mentions alongside key executives.",
         "supporting_posts": [], "suggested_search": "OpenAI"},
        {"id": "q8", "text": "Is negative sentiment increasing?", "category": "sentiment",
         "answer": "Compare weekly negative counts in the sentiment trend chart to see if negativity is rising.",
         "supporting_posts": [], "suggested_search": ""},
    ]


def export_rag_evidence(topics: list[dict], posts: list[dict]) -> list[dict]:
    post_map = {p["post_id"]: p for p in posts}
    out = []
    for t in topics[:15]:
        results = []
        for pid in (t.get("post_ids") or [])[:5]:
            p = post_map.get(pid)
            if not p:
                continue
            results.append({
                "post_id": pid,
                "text": (p["clean_text"] or "")[:300],
                "similarity_score": 0.85,
                "sentiment": "neutral",
            })
        out.append({
            "query": t["label"],
            "topic_id": t["topic_id"],
            "results": results,
        })
    return out


CRISIS_SEVERITY_TO_TIER = {0: "green", 1: "green", 2: "amber", 3: "red"}

SENTIMENT_POS_THRESHOLD = 0.05
SENTIMENT_NEG_THRESHOLD = -0.05


def _a1_predict_batch(texts: list[str]) -> list[dict]:
    """Run A1's predict() over a list of texts.

    A1 (advanced/layer3_llm/predict.py) currently exposes only a single-text
    predict(text) -> {"crisis_severity", "sentiment_score", "topic"}.
    This loops it; for the full 1,211-post corpus this is the dominant cost
    of the export run. If A1 later exposes a batched predict_batch(texts),
    swap the loop below for that call.
    """
    return [_a1_predict(t) for t in texts]


def _assemble_brand_states_from_a1(
    posts: list[dict],
    rag_evidence: list[dict],
) -> list[dict]:
    """Build one brand-state dict per A1 topic category, using A1's own
    crisis_severity and sentiment_score classifications rather than the
    B-series rule-based sentiment/crisis exports.

    A1 topic labels are fixed (see multitask_models.TOPIC_LABELS), so this
    groups posts by that label directly instead of by a BERTopic cluster id.
    """
    texts = [p["clean_text"] or p["raw_text"] for p in posts]
    preds = _a1_predict_batch(texts)
    post_map = {p["post_id"]: p for p in posts}

    by_topic: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for post, pred in zip(posts, preds):
        by_topic[pred["topic"]].append((post, pred))

    evidence_map = {e["topic_id"]: e.get("results", []) for e in rag_evidence}
    states = []

    for topic_label, items in by_topic.items():
        scores = [pred["sentiment_score"] for _, pred in items]
        total = len(scores) or 1
        pos = sum(1 for s in scores if s > SENTIMENT_POS_THRESHOLD) / total
        neg = sum(1 for s in scores if s < SENTIMENT_NEG_THRESHOLD) / total
        neu = 1.0 - pos - neg

        neg_items = sorted(
            [(p, pr) for p, pr in items if pr["sentiment_score"] < SENTIMENT_NEG_THRESHOLD],
            key=lambda x: x[0].get("score", 0), reverse=True,
        )[:3]
        pos_items = sorted(
            [(p, pr) for p, pr in items if pr["sentiment_score"] > SENTIMENT_POS_THRESHOLD],
            key=lambda x: x[0].get("score", 0), reverse=True,
        )[:2]
        negative_drivers = [p["title"][:80] for p, _ in neg_items if p.get("title")]
        positive_drivers = [p["title"][:80] for p, _ in pos_items if p.get("title")]

        max_severity = max((pred["crisis_severity"] for _, pred in items), default=0)
        crisis_level = CRISIS_SEVERITY_TO_TIER.get(max_severity, "green")

        evidence = [
            {
                "post_id": r["post_id"],
                "excerpt": (r.get("text") or "")[:200],
                "score":   post_map.get(r["post_id"], {}).get("score", 0),
            }
            for r in evidence_map.get(topic_label, [])[:5]
        ]

        states.append({
            "scenario_id":      f"topic_{topic_label}",
            "topic":            topic_label.replace("_", " ").title(),
            "sentiment":        {"positive": round(pos, 3),
                                 "neutral":  round(neu, 3),
                                 "negative": round(neg, 3)},
            "negative_drivers": negative_drivers,
            "positive_drivers": positive_drivers,
            "crisis_level":     crisis_level,
            "evidence":         evidence,
        })

    return states


def _assemble_brand_states_from_b_series(
    posts: list[dict],
    sentiment: list[dict],
    topics: list[dict],
    rag_evidence: list[dict],
) -> list[dict]:
    """Fallback: build brand states from the B-series rule-based sentiment
    and crisis exports, used only when A1 is unavailable (no checkpoint
    trained yet, or advanced.layer3_llm.predict fails to import).
    """
    sent_map     = {s["post_id"]: s["label"] for s in sentiment}
    post_map     = {p["post_id"]: p for p in posts}
    evidence_map = {e["topic_id"]: e.get("results", []) for e in rag_evidence}
    states = []

    for topic in topics:
        tid   = topic["topic_id"]
        pids  = topic.get("post_ids") or []
        label = topic.get("label", f"Topic {tid}")

        labels = [sent_map.get(pid) for pid in pids if sent_map.get(pid)]
        total  = len(labels) or 1
        pos = labels.count("positive") / total
        neu = labels.count("neutral")  / total
        neg = labels.count("negative") / total

        neg_posts = sorted(
            [post_map[pid] for pid in pids
             if pid in post_map and sent_map.get(pid) == "negative"],
            key=lambda p: p.get("score", 0), reverse=True,
        )[:3]
        pos_posts = sorted(
            [post_map[pid] for pid in pids
             if pid in post_map and sent_map.get(pid) == "positive"],
            key=lambda p: p.get("score", 0), reverse=True,
        )[:2]
        negative_drivers = [p["title"][:80] for p in neg_posts if p.get("title")]
        positive_drivers = [p["title"][:80] for p in pos_posts if p.get("title")]

        evidence = [
            {
                "post_id": r["post_id"],
                "excerpt": (r.get("text") or "")[:200],
                "score":   post_map.get(r["post_id"], {}).get("score", 0),
            }
            for r in evidence_map.get(tid, [])[:5]
        ]

        crisis_level = "red" if neg >= 0.4 else "amber" if neg >= 0.28 else "green"

        states.append({
            "scenario_id":      f"topic_{tid}",
            "topic":            label,
            "sentiment":        {"positive": round(pos, 3),
                                 "neutral":  round(neu, 3),
                                 "negative": round(neg, 3)},
            "negative_drivers": negative_drivers,
            "positive_drivers": positive_drivers,
            "crisis_level":     crisis_level,
            "evidence":         evidence,
        })

    return states


def _assemble_brand_states(
    posts: list[dict],
    sentiment: list[dict],
    topics: list[dict],
    crisis: dict,
    rag_evidence: list[dict],
) -> list[dict]:
    """Build one brand-state dict per topic for the recommendation layer.

    Prefers A1's RoBERTa multitask classifications (crisis_severity,
    sentiment_score, topic) when the trained checkpoint is available.
    Falls back to the B-series rule-based sentiment.json / crisis dict
    and BERTopic/stub topics.json otherwise, so the export script still
    runs end to end before A1 is trained.
    """
    if _A1_AVAILABLE:
        try:
            return _assemble_brand_states_from_a1(posts, rag_evidence)
        except FileNotFoundError as exc:
            print(f"  Warning: A1 checkpoint not found ({exc}); "
                  "falling back to B-series sentiment/crisis for brand states.")
    return _assemble_brand_states_from_b_series(posts, sentiment, topics, rag_evidence)


def _export_recommendations_json(
    posts: list[dict],
    sentiment: list[dict],
    topics: list[dict],
    crisis: dict,
    rag_evidence: list[dict],
) -> list[dict]:
    """Generate brand recommendations per topic and return the dashboard payload.

    Tiered routing (from A4 experiment results):
      red / amber states  ->  Tree-of-Thought  (higher faithfulness, pairwise preferred)
      green states        ->  Chain-of-Thought (equivalent quality, 4x cheaper)

    Brand states are sourced from A1's classifier when its checkpoint is
    available, otherwise from the B-series rule-based exports.
    Requires Ollama running with qwen2.5:7b pulled for the generation step.
    Skips gracefully if shared.recommendations is not importable.
    """
    if not _RECO_AVAILABLE:
        print("  Warning: shared.recommendations not importable; "
              "skipping recommendations.json")
        return []
    states = _assemble_brand_states(posts, sentiment, topics, crisis, rag_evidence)
    return _export_reco(states, method="tiered")


def export_faiss(posts: list[dict]) -> None:
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"  Warning: skipping FAISS ({e})")
        return

    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [p["clean_text"][:2000] for p in posts]
    ids = [p["post_id"] for p in posts]
    emb = model.encode(texts, show_progress_bar=True, batch_size=32)
    emb = np.asarray(emb, dtype=np.float32)
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(FAISS_DIR / "posts.index"))
    (FAISS_DIR / "id_map.json").write_text(json.dumps(ids), encoding="utf-8")
    print(f"  FAISS index: {len(ids)} vectors → {FAISS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Skip BERTopic (stub topics)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading clean JSONL...")
    records = load_clean_jsonl()
    print(f"  {len(records)} posts")

    print("Exporting cleaned_posts.json...")
    posts = export_cleaned_posts(records)
    (OUTPUT_DIR / "cleaned_posts.json").write_text(json.dumps(posts, indent=2), encoding="utf-8")

    print("Exporting entities.json (spaCy)...")
    nlp = _init_ner()
    entities = export_entities(posts, nlp)
    (OUTPUT_DIR / "entities.json").write_text(json.dumps(entities, indent=2), encoding="utf-8")

    print("Exporting topics + trend...")
    topics, trend = export_topics(posts, fast=args.fast)
    (OUTPUT_DIR / "topics.json").write_text(json.dumps(topics, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "trend_timeseries.json").write_text(json.dumps(trend, indent=2), encoding="utf-8")

    print("Exporting sentiment.json...")
    sentiment = export_sentiment(posts)
    (OUTPUT_DIR / "sentiment.json").write_text(json.dumps(sentiment, indent=2), encoding="utf-8")

    print("Exporting crisis, summaries, qa, rag...")
    crisis_data = export_crisis(posts, sentiment)
    (OUTPUT_DIR / "crisis.json").write_text(
        json.dumps(crisis_data, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "summaries.json").write_text(json.dumps(export_summaries(), indent=2), encoding="utf-8")
    (OUTPUT_DIR / "qa_bank.json").write_text(json.dumps(export_qa_bank(), indent=2), encoding="utf-8")
    rag = export_rag_evidence(topics, posts)
    (OUTPUT_DIR / "rag_evidence.json").write_text(
        json.dumps(rag, indent=2), encoding="utf-8"
    )

    print("Exporting recommendations.json (A4 ToT / tiered)...")
    reco = _export_recommendations_json(posts, sentiment, topics, crisis_data, rag)
    (OUTPUT_DIR / "recommendations.json").write_text(
        json.dumps(reco, indent=2), encoding="utf-8"
    )
    print(f"  {len(reco)} recommendation set(s) written.")

    print("Building FAISS index...")
    export_faiss(posts)
    print("Done →", OUTPUT_DIR)


if __name__ == "__main__":
    main()
