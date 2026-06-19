"""Thin wrapper around the local Ollama HTTP API (no API key required)."""

from __future__ import annotations

import json
import os
from typing import Any

import requests

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


def ollama_available(base_url: str | None = None, timeout: float = 2.0) -> bool:
    base_url = base_url or DEFAULT_OLLAMA_URL
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        return response.status_code == 200
    except (requests.RequestException, OSError):
        return False


def query_ollama(
    prompt: str,
    *,
    system: str = "",
    model: str | None = None,
    base_url: str | None = None,
    json_mode: bool = True,
    timeout: float = 300.0,
) -> str:
    """Send a chat completion to a local Ollama model."""
    base_url = (base_url or DEFAULT_OLLAMA_URL).rstrip("/")
    model = model or DEFAULT_OLLAMA_MODEL
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0},
    }
    if json_mode:
        payload["format"] = "json"

    response = requests.post(
        f"{base_url}/api/chat",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return str(data.get("message", {}).get("content", ""))
