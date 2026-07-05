"""Chunking metadata helpers for trainable dataset artifacts."""

from __future__ import annotations


def chunking_profile(processed: dict[str, object]) -> dict[str, object]:
    operations = [str(op) for op in processed.get("operations", [])] if isinstance(processed.get("operations"), list) else []
    return {
        "enabled": any("chunk" in op.lower() for op in operations),
        "chunk_manifest_ref": str(processed.get("chunk_manifest_ref", "")).strip() or None,
        "min_tokens": int(processed.get("min_tokens", 256) or 256),
    }
