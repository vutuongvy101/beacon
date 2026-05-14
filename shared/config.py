"""Model config, runtime flags, and path helpers — no secrets stored here."""

from __future__ import annotations

import os
from pathlib import Path

# ── runtime flags ─────────────────────────────────────────────────────────────

# Set DEMO_MODE=true when serving the marker demo from cached snapshots only.
DEMO_MODE: bool = os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")

# Optional: override data root (e.g. Colab mount path)
DATA_DIR: str | None = os.getenv("DATA_DIR") or None

# ── Qwen via Ollama (local, no API key required) ──────────────────────────────

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Fast model — demo, classification tasks, caching (A1 Part 2, A5)
QWEN_SMALL: str = os.getenv("QWEN_SMALL", "qwen2.5:3b")

# Capable model — reasoning chains, report generation (A3 CoT/ReAct, Report Agent)
QWEN_LARGE: str = os.getenv("QWEN_LARGE", "qwen2.5:7b")

# Vision-language model — multimodal analysis (A7); run on Colab T4
QWEN_VL: str = os.getenv("QWEN_VL", "qwen2-vl:7b")

# ── path helpers ──────────────────────────────────────────────────────────────

def get_project_root() -> Path:
    """Repository root (parent of ``shared/``)."""
    return Path(__file__).resolve().parent.parent


def get_data_dir() -> Path:
    """Directory containing ``sample.csv``, ``snapshots/``, ``faiss_index/``."""
    if DATA_DIR:
        return Path(DATA_DIR).expanduser().resolve()
    return get_project_root() / "data"
