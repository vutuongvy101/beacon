"""Q-Former-style and standard multi-task RoBERTa architectures."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel

TOPIC_LABELS = [
    "product_releases",
    "pricing_subscriptions",
    "api_developer",
    "safety_ethics",
    "corporate_leadership",
    "competition",
    "reliability_outages",
    "general_discussion",
]

NUM_CRISIS = 4
NUM_QUERIES = 6
NUM_XATTN_LAYERS = 2
LOSS_WEIGHTS = {"crisis": 1.0, "sentiment": 0.5, "topic": 0.5}


class CrossAttentionBlock(nn.Module):
    """One cross-attention layer: task queries attend to encoder hidden states."""

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_size)
        self.norm_kv = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(hidden_size)

    def forward(
        self,
        queries: torch.Tensor,
        encoder_hidden: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Update queries via cross-attention over encoder tokens."""
        q = self.norm_q(queries)
        kv = self.norm_kv(encoder_hidden)
        attn_out, _ = self.attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        queries = queries + attn_out
        queries = queries + self.ffn(self.norm_ffn(queries))
        return queries


class TaskQueryTower(nn.Module):
    """Learnable query tokens + stacked cross-attention for one task."""

    def __init__(
        self,
        hidden_size: int,
        num_queries: int = NUM_QUERIES,
        num_layers: int = NUM_XATTN_LAYERS,
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_size) * 0.02)
        self.layers = nn.ModuleList(
            [CrossAttentionBlock(hidden_size) for _ in range(num_layers)]
        )

    def forward(
        self,
        encoder_hidden: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = encoder_hidden.size(0)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        for layer in self.layers:
            queries = layer(queries, encoder_hidden, key_padding_mask)
        return queries.mean(dim=1)


def _freeze_encoder(encoder: RobertaModel) -> None:
    for param in encoder.parameters():
        param.requires_grad = False


def _task_losses(
    crisis_logits: torch.Tensor,
    sentiment_pred: torch.Tensor,
    topic_logits: torch.Tensor,
    crisis_labels: torch.Tensor,
    sentiment_labels: torch.Tensor,
    topic_labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    loss_crisis = F.cross_entropy(crisis_logits, crisis_labels)
    loss_sentiment = F.mse_loss(sentiment_pred.squeeze(-1), sentiment_labels.float())
    loss_topic = F.cross_entropy(topic_logits, topic_labels)
    total = (
        LOSS_WEIGHTS["crisis"] * loss_crisis
        + LOSS_WEIGHTS["sentiment"] * loss_sentiment
        + LOSS_WEIGHTS["topic"] * loss_topic
    )
    return {
        "loss": total,
        "loss_crisis": loss_crisis,
        "loss_sentiment": loss_sentiment,
        "loss_topic": loss_topic,
    }


class QFormerMultiTaskRoberta(nn.Module):
    """Frozen RoBERTa encoder + per-task Q-Former query towers."""

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_topics: int = len(TOPIC_LABELS),
        num_queries: int = NUM_QUERIES,
        num_xattn_layers: int = NUM_XATTN_LAYERS,
    ) -> None:
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        _freeze_encoder(self.encoder)

        self.crisis_tower = TaskQueryTower(hidden_size, num_queries, num_xattn_layers)
        self.sentiment_tower = TaskQueryTower(hidden_size, num_queries, num_xattn_layers)
        self.topic_tower = TaskQueryTower(hidden_size, num_queries, num_xattn_layers)

        self.crisis_head = nn.Linear(hidden_size, NUM_CRISIS)
        self.sentiment_head = nn.Linear(hidden_size, 1)
        self.topic_head = nn.Linear(hidden_size, num_topics)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        key_padding_mask = attention_mask == 0
        return hidden, key_padding_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        crisis_labels: torch.Tensor | None = None,
        sentiment_labels: torch.Tensor | None = None,
        topic_labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        hidden, key_padding_mask = self.encode(input_ids, attention_mask)

        crisis_vec = self.crisis_tower(hidden, key_padding_mask)
        sentiment_vec = self.sentiment_tower(hidden, key_padding_mask)
        topic_vec = self.topic_tower(hidden, key_padding_mask)

        crisis_logits = self.crisis_head(crisis_vec)
        sentiment_pred = self.sentiment_head(sentiment_vec)
        topic_logits = self.topic_head(topic_vec)

        out: dict[str, Any] = {
            "crisis_logits": crisis_logits,
            "sentiment_pred": sentiment_pred,
            "topic_logits": topic_logits,
        }

        if crisis_labels is not None and sentiment_labels is not None and topic_labels is not None:
            losses = _task_losses(
                crisis_logits,
                sentiment_pred,
                topic_logits,
                crisis_labels,
                sentiment_labels,
                topic_labels,
            )
            out.update(losses)

        return out

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class StandardMultiTaskRoberta(nn.Module):
    """Frozen RoBERTa + shared [CLS] vector → three task heads (baseline)."""

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_topics: int = len(TOPIC_LABELS),
    ) -> None:
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        _freeze_encoder(self.encoder)

        self.crisis_head = nn.Linear(hidden_size, NUM_CRISIS)
        self.sentiment_head = nn.Linear(hidden_size, 1)
        self.topic_head = nn.Linear(hidden_size, num_topics)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        crisis_labels: torch.Tensor | None = None,
        sentiment_labels: torch.Tensor | None = None,
        topic_labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls_vec = hidden[:, 0]

        crisis_logits = self.crisis_head(cls_vec)
        sentiment_pred = self.sentiment_head(cls_vec)
        topic_logits = self.topic_head(cls_vec)

        out: dict[str, Any] = {
            "crisis_logits": crisis_logits,
            "sentiment_pred": sentiment_pred,
            "topic_logits": topic_logits,
        }

        if crisis_labels is not None and sentiment_labels is not None and topic_labels is not None:
            losses = _task_losses(
                crisis_logits,
                sentiment_pred,
                topic_logits,
                crisis_labels,
                sentiment_labels,
                topic_labels,
            )
            out.update(losses)

        return out

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
