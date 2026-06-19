"""Training and evaluation helpers for Layer 3 multi-task models."""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup

from advanced.layer3_llm.multitask_models import TOPIC_LABELS


class MultiTaskDataset(Dataset):
    """Tokenised multi-task dataset."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: RobertaTokenizer,
        text_col: str = "text",
        max_length: int = 256,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.max_length = max_length
        self.topic_to_id = {t: i for i, t in enumerate(TOPIC_LABELS)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            str(row[self.text_col]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        topic = str(row["topic"])
        if topic not in self.topic_to_id:
            topic = "general_discussion"
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "crisis_labels": torch.tensor(int(row["crisis_severity"]), dtype=torch.long),
            "sentiment_labels": torch.tensor(float(row["sentiment_score"]), dtype=torch.float),
            "topic_labels": torch.tensor(self.topic_to_id[topic], dtype=torch.long),
        }


def get_device() -> torch.device:
    """Pick compute device; default to CPU on macOS (MPS has checkpoint reload bugs)."""
    override = os.getenv("LAYER3_DEVICE", "").lower()
    if override in ("cpu", "cuda", "mps"):
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 3,
    lr: float = 2e-4,
    device: torch.device | None = None,
) -> tuple[dict[str, list[float]], float]:
    """Train a multi-task model; return history and wall-clock seconds."""
    device = device or get_device()
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_loss_crisis": [],
        "val_loss_crisis": [],
        "train_loss_sentiment": [],
        "val_loss_sentiment": [],
        "train_loss_topic": [],
        "val_loss_topic": [],
    }

    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        train_totals = {"loss": 0.0, "loss_crisis": 0.0, "loss_sentiment": 0.0, "loss_topic": 0.0}
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} train", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            out["loss"].backward()
            optimizer.step()
            scheduler.step()
            for key in train_totals:
                train_totals[key] += out[key].item()

        n_train = len(train_loader)
        for key in train_totals:
            history[f"train_{key}" if key != "loss" else "train_loss"].append(train_totals[key] / n_train)

        model.eval()
        val_totals = {"loss": 0.0, "loss_crisis": 0.0, "loss_sentiment": 0.0, "loss_topic": 0.0}
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} val", leave=False):
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                for key in val_totals:
                    val_totals[key] += out[key].item()

        n_val = max(len(val_loader), 1)
        for key in val_totals:
            history[f"val_{key}" if key != "loss" else "val_loss"].append(val_totals[key] / n_val)

        print(
            f"Epoch {epoch + 1}: train_loss={history['train_loss'][-1]:.4f} "
            f"val_loss={history['val_loss'][-1]:.4f} "
            f"(crisis={history['val_loss_crisis'][-1]:.4f}, "
            f"sent={history['val_loss_sentiment'][-1]:.4f}, "
            f"topic={history['val_loss_topic'][-1]:.4f})"
        )

    return history, time.perf_counter() - t0


@torch.inference_mode()
def predict_batch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device | None = None,
) -> dict[str, list]:
    device = device or get_device()
    model = model.to(device)
    model.eval()

    crisis_preds: list[int] = []
    sentiment_preds: list[float] = []
    topic_preds: list[int] = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k.endswith("labels") is False}
        out = model(**batch)
        crisis_preds.extend(out["crisis_logits"].argmax(dim=-1).cpu().tolist())
        sentiment_preds.extend(out["sentiment_pred"].squeeze(-1).cpu().tolist())
        topic_preds.extend(out["topic_logits"].argmax(dim=-1).cpu().tolist())

    return {
        "crisis_pred": crisis_preds,
        "sentiment_pred": sentiment_preds,
        "topic_pred": topic_preds,
    }


def evaluate_predictions(
    y_crisis_true: list[int],
    y_crisis_pred: list[int],
    y_sent_true: list[float],
    y_sent_pred: list[float],
    y_topic_true: list[int],
    y_topic_pred: list[int],
) -> dict[str, Any]:
    crisis_f1 = f1_score(y_crisis_true, y_crisis_pred, average="macro", zero_division=0)
    topic_f1 = f1_score(y_topic_true, y_topic_pred, average="macro", zero_division=0)
    sent_mae = mean_absolute_error(y_sent_true, y_sent_pred)
    sent_rmse = float(np.sqrt(mean_squared_error(y_sent_true, y_sent_pred)))

    return {
        "crisis_accuracy": accuracy_score(y_crisis_true, y_crisis_pred),
        "crisis_f1_macro": crisis_f1,
        "crisis_confusion_matrix": confusion_matrix(
            y_crisis_true, y_crisis_pred, labels=[0, 1, 2, 3]
        ).tolist(),
        "sentiment_mae": sent_mae,
        "sentiment_rmse": sent_rmse,
        "topic_accuracy": accuracy_score(y_topic_true, y_topic_pred),
        "topic_f1_macro": topic_f1,
        "avg_score": float(np.mean([crisis_f1, topic_f1, 1.0 - sent_mae / 2.0])),
    }


def save_checkpoint(
    path,
    model: torch.nn.Module,
    arch: str,
    config: dict[str, Any],
    history: dict[str, list[float]] | None = None,
) -> None:
    torch.save(
        {
            "arch": arch,
            "state_dict": model.state_dict(),
            "topic_labels": TOPIC_LABELS,
            "config": config,
            "train_metrics": history or {},
        },
        path,
    )


def load_checkpoint(path, device: torch.device | None = None):
    device = device or get_device()
    return torch.load(path, map_location=device, weights_only=False)


def measure_inference_latency_ms(
    model: torch.nn.Module,
    sample_loader: DataLoader,
    device: torch.device | None = None,
    n_posts: int = 50,
) -> float:
    device = device or get_device()
    model = model.to(device)
    model.eval()
    times: list[float] = []
    count = 0
    with torch.inference_mode():
        for batch in sample_loader:
            batch = {k: v.to(device) for k, v in batch.items() if not k.endswith("labels")}
            t0 = time.perf_counter()
            model(**batch)
            elapsed = (time.perf_counter() - t0) / len(batch["input_ids"])
            times.extend([elapsed] * len(batch["input_ids"]))
            count += len(batch["input_ids"])
            if count >= n_posts:
                break
    return float(np.mean(times[:n_posts]) * 1000)
