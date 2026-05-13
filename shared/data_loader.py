"""Data loading helpers — used by all notebooks and pipeline modules.

Public API
----------
load_reddit(brand, date, as_df)   Load a single day's Reddit snapshot.
load_all_reddit(brand, as_df)     Merge all Reddit snapshots, deduplicating by post_id.
load_sample(brand, as_df)         Alias for load_all_reddit — matches interface expected
                                  by downstream notebooks.
"""

from __future__ import annotations

import json
from datetime import date as _date
from pathlib import Path
from typing import Any

from shared.config import get_data_dir

# ── internal helpers ──────────────────────────────────────────────────────────

def _snapshot_dir() -> Path:
    return get_data_dir() / "snapshots"


def _iter_jsonl(path: Path):
    """Yield parsed dicts from a JSONL file, skipping blank/malformed lines."""
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  Warning: skipping malformed line {lineno} in {path.name}: {exc}")


def _to_df(records: list[dict[str, Any]]):
    """Convert records to a pandas DataFrame (imported lazily)."""
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required for as_df=True — run: pip install pandas") from e

    df = pd.DataFrame(records)
    if "created_utc" in df.columns:
        df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True, errors="coerce")
    return df

# ── public API ────────────────────────────────────────────────────────────────

def load_reddit(
    brand: str = "openai",
    snapshot_date: _date | str | None = None,
    as_df: bool = True,
) -> Any:
    """Load a single Reddit snapshot.

    Parameters
    ----------
    brand:
        Brand label used in the snapshot filename (e.g. ``"openai"``).
    snapshot_date:
        Date of the snapshot to load.  Accepts a ``datetime.date``, an ISO
        string (``"2026-05-14"``), or ``None`` (loads the most recent file).
    as_df:
        If ``True`` (default), return a ``pandas.DataFrame``.
        If ``False``, return a ``list[dict]``.
    """
    sdir = _snapshot_dir()

    if snapshot_date is None:
        candidates = sorted(sdir.glob(f"reddit_{brand}_????????.jsonl"))
        if not candidates:
            raise FileNotFoundError(
                f"No Reddit snapshot found for brand '{brand}' in {sdir}"
            )
        path = candidates[-1]
    else:
        if isinstance(snapshot_date, str):
            snapshot_date = _date.fromisoformat(snapshot_date)
        path = sdir / f"reddit_{brand}_{snapshot_date:%Y%m%d}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Snapshot not found: {path}")

    records = list(_iter_jsonl(path))
    print(f"Loaded {len(records)} posts from {path.name}")
    return _to_df(records) if as_df else records


def load_all_reddit(
    brand: str = "openai",
    as_df: bool = True,
) -> Any:
    """Load and merge all Reddit snapshots for a brand, deduplicating by post_id.

    Parameters
    ----------
    brand:
        Brand label used in snapshot filenames (e.g. ``"openai"``).
    as_df:
        If ``True`` (default), return a ``pandas.DataFrame``.
        If ``False``, return a ``list[dict]``.
    """
    sdir = _snapshot_dir()
    paths = sorted(sdir.glob(f"reddit_{brand}_????????.jsonl"))
    if not paths:
        raise FileNotFoundError(
            f"No Reddit snapshots found for brand '{brand}' in {sdir}"
        )

    seen_ids: set[str] = set()
    records: list[dict] = []
    for path in paths:
        before = len(records)
        for rec in _iter_jsonl(path):
            pid = rec.get("post_id", "")
            if pid and pid not in seen_ids:
                records.append(rec)
                seen_ids.add(pid)
        print(f"  {path.name}: +{len(records) - before} posts")

    print(f"Total: {len(records)} unique posts across {len(paths)} snapshot(s)")
    return _to_df(records) if as_df else records


def load_sample(
    brand: str = "openai",
    as_df: bool = True,
) -> Any:
    """Load all available Reddit posts for a brand (deduped across snapshots).

    This is the primary entry point used by notebooks and pipeline modules.
    It is equivalent to ``load_all_reddit(brand, as_df)``.
    """
    return load_all_reddit(brand=brand, as_df=as_df)
