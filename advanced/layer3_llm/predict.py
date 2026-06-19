"""Phase 2 export — run trained Q-Former multi-task model on a single post."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import RobertaTokenizer

from advanced.layer3_llm.multitask_models import QFormerMultiTaskRoberta, TOPIC_LABELS
from advanced.layer3_llm.train_utils import get_device

_MODEL: QFormerMultiTaskRoberta | None = None
_TOKENIZER: RobertaTokenizer | None = None
_TOPIC_LABELS: list[str] = list(TOPIC_LABELS)
_MAX_LENGTH: int = 256
_DEVICE: torch.device | None = None


def _checkpoint_path() -> Path:
    return Path(__file__).resolve().parent / "outputs" / "qformer_model.pt"


def _load_model() -> None:
    global _MODEL, _TOKENIZER, _TOPIC_LABELS, _MAX_LENGTH, _DEVICE

    if _MODEL is not None:
        return

    ckpt_path = _checkpoint_path()
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Q-Former checkpoint not found: {ckpt_path}\n"
            "Run advanced/A1_redeveloped_llm.ipynb through Section 3 first."
        )

    _DEVICE = get_device()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    _MAX_LENGTH = int(config.get("max_length", 256))
    _TOPIC_LABELS = ckpt.get("topic_labels", TOPIC_LABELS)

    _TOKENIZER = RobertaTokenizer.from_pretrained(config.get("model_name", "roberta-base"))
    _MODEL = QFormerMultiTaskRoberta(
        model_name=config.get("model_name", "roberta-base"),
        num_topics=len(_TOPIC_LABELS),
        num_queries=int(config.get("num_queries", 6)),
        num_xattn_layers=int(config.get("num_xattn_layers", 2)),
    )
    _MODEL.load_state_dict(ckpt["state_dict"])
    _MODEL.to(_DEVICE)
    _MODEL.eval()


def predict(text: str) -> dict:
    """
    Run the trained Q-Former multi-task model on a single Reddit post.

    Args:
        text: Raw or cleaned post text.

    Returns:
        dict with keys:
            "crisis_severity": int, 0-3
            "sentiment_score": float, -1.0 to 1.0
            "topic": str, one of the fixed topic categories
    """
    _load_model()
    assert _MODEL is not None and _TOKENIZER is not None and _DEVICE is not None

    enc = _TOKENIZER(
        text,
        truncation=True,
        padding="max_length",
        max_length=_MAX_LENGTH,
        return_tensors="pt",
    )
    enc = {k: v.to(_DEVICE) for k, v in enc.items()}

    with torch.inference_mode():
        out = _MODEL(**enc)

    crisis = int(out["crisis_logits"].argmax(dim=-1).item())
    sentiment = float(out["sentiment_pred"].squeeze().item())
    sentiment = max(-1.0, min(1.0, sentiment))
    topic_idx = int(out["topic_logits"].argmax(dim=-1).item())
    topic = _TOPIC_LABELS[topic_idx]

    return {
        "crisis_severity": crisis,
        "sentiment_score": round(sentiment, 4),
        "topic": topic,
    }
