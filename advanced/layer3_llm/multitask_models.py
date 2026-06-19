"""Q-Former-style, standard, and LoRA multi-task RoBERTa architectures."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
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
LOSS_WEIGHTS = {"crisis": 1.5, "sentiment": 4.0, "topic": 0.25}
LABEL_SMOOTHING = 0.05


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


def _set_encoder_trainable(encoder: RobertaModel, *, unfreeze_last_n: int = 0) -> None:
    _freeze_encoder(encoder)
    if unfreeze_last_n <= 0:
        return
    layers = encoder.encoder.layer
    for layer in layers[-unfreeze_last_n:]:
        for param in layer.parameters():
            param.requires_grad = True


def _task_losses(
    crisis_logits: torch.Tensor,
    sentiment_pred: torch.Tensor,
    topic_logits: torch.Tensor,
    crisis_labels: torch.Tensor,
    sentiment_labels: torch.Tensor,
    topic_labels: torch.Tensor,
    *,
    loss_weights: dict[str, float],
    crisis_class_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    loss_crisis = F.cross_entropy(
        crisis_logits,
        crisis_labels,
        weight=crisis_class_weights,
        label_smoothing=LABEL_SMOOTHING,
    )
    loss_sentiment = F.mse_loss(sentiment_pred.squeeze(-1), sentiment_labels.float())
    loss_topic = F.cross_entropy(topic_logits, topic_labels, label_smoothing=LABEL_SMOOTHING)
    total = (
        loss_weights["crisis"] * loss_crisis
        + loss_weights["sentiment"] * loss_sentiment
        + loss_weights["topic"] * loss_topic
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
        crisis_class_weights: torch.Tensor | None = None,
        loss_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        _freeze_encoder(self.encoder)
        if crisis_class_weights is not None:
            self.register_buffer("crisis_class_weights", crisis_class_weights)
        else:
            self.crisis_class_weights = None
        self.loss_weights = dict(loss_weights or LOSS_WEIGHTS)

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
                loss_weights=self.loss_weights,
                crisis_class_weights=self.crisis_class_weights,
            )
            out.update(losses)

        return out

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_loss_weights(self, loss_weights: dict[str, float]) -> None:
        self.loss_weights = dict(loss_weights)


class StandardMultiTaskRoberta(nn.Module):
    """Frozen RoBERTa + shared [CLS] vector → three task heads (baseline)."""

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_topics: int = len(TOPIC_LABELS),
        unfreeze_last_n: int = 2,
        crisis_class_weights: torch.Tensor | None = None,
        loss_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        _set_encoder_trainable(self.encoder, unfreeze_last_n=unfreeze_last_n)
        if crisis_class_weights is not None:
            self.register_buffer("crisis_class_weights", crisis_class_weights)
        else:
            self.crisis_class_weights = None
        self.loss_weights = dict(loss_weights or LOSS_WEIGHTS)

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
                loss_weights=self.loss_weights,
                crisis_class_weights=self.crisis_class_weights,
            )
            out.update(losses)

        return out

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_loss_weights(self, loss_weights: dict[str, float]) -> None:
        self.loss_weights = dict(loss_weights)


# ── LoRA variant ──────────────────────────────────────────────────────────────

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1


class LoRAMultiTaskRoberta(nn.Module):
    """RoBERTa with LoRA adapters on attention layers + multi-task heads.

    Injects low-rank adapters (ΔW = BA) into the query and value projection
    matrices of every encoder layer.  All base encoder weights stay frozen;
    only the LoRA matrices and task heads are trained — typically ~300 K
    trainable parameters vs ~14 M for ``StandardMultiTaskRoberta``.
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_topics: int = len(TOPIC_LABELS),
        lora_r: int = LORA_R,
        lora_alpha: int = LORA_ALPHA,
        lora_dropout: float = LORA_DROPOUT,
        crisis_class_weights: torch.Tensor | None = None,
        loss_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        base_encoder = RobertaModel.from_pretrained(model_name)
        hidden_size = base_encoder.config.hidden_size

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "value"],
        )
        self.encoder = get_peft_model(base_encoder, lora_config)

        if crisis_class_weights is not None:
            self.register_buffer("crisis_class_weights", crisis_class_weights)
        else:
            self.crisis_class_weights = None
        self.loss_weights = dict(loss_weights or LOSS_WEIGHTS)

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
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
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
                loss_weights=self.loss_weights,
                crisis_class_weights=self.crisis_class_weights,
            )
            out.update(losses)

        return out

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_loss_weights(self, w: dict[str, float]) -> None:
        self.loss_weights = dict(w)

    def merged_state_dict(self) -> dict[str, Any]:
        """Merge LoRA weights into base encoder and return a flat state dict.

        Uses a deep copy so the original model stays intact for further use.
        The merged checkpoint can be loaded by ``StandardMultiTaskRoberta``
        (with ``unfreeze_last_n=0``) so ``predict.py`` stays unchanged.
        """
        import copy

        encoder_copy = copy.deepcopy(self.encoder)
        merged_encoder = encoder_copy.merge_and_unload()
        state: dict[str, Any] = {}
        for k, v in merged_encoder.state_dict().items():
            state[f"encoder.{k}"] = v
        for name in ("crisis_head", "sentiment_head", "topic_head"):
            head = getattr(self, name)
            for k, v in head.state_dict().items():
                state[f"{name}.{k}"] = v
        return state
