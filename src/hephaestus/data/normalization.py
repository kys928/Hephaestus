"""Deterministic preprocessing normalization metadata."""

from __future__ import annotations


def normalize_operations(processed: dict[str, object]) -> list[str]:
    raw = processed.get("operations", [])
    if not isinstance(raw, list | tuple):
        return []
    return [str(op).strip() for op in raw if str(op).strip()]


def normalization_profile(processed: dict[str, object]) -> dict[str, object]:
    operations = normalize_operations(processed)
    return {"enabled": "normalize" in operations or "normalization" in operations, "operations": operations}
