from __future__ import annotations

from typing import Any


def stable_chunk_id(document_name: str, chunk_index: int, seed: str | None = None) -> str:
    base = f"{document_name}:{chunk_index}"
    if seed:
        base = f"{base}:{seed}"
    return base.replace(" ", "_").replace("/", "_")


def normalize_backend_name(value: str) -> str:
    return value.strip().lower()


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
