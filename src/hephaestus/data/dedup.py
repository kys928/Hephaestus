"""Deduplication metadata helpers that avoid embedding example data."""

from __future__ import annotations


def deduplication_profile(processed: dict[str, object]) -> dict[str, object]:
    operations = [str(op) for op in processed.get("operations", [])] if isinstance(processed.get("operations"), list) else []
    return {
        "enabled": any("dedup" in op.lower() for op in operations),
        "dropped_examples": int(processed.get("dropped_examples", 0) or 0),
        "evidence_ref": str(processed.get("dedup_report_ref", "")).strip() or None,
    }
