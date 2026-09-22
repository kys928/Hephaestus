"""Deterministic normalization for untrusted model-to-model handoffs.

Normalization is intentionally semantic-neutral: it parses and envelopes a producer
record, but it never translates producer values into the consumer's contract.  The
consumer still owns interpretation; deterministic guards still own execution safety.
"""
from __future__ import annotations

import json
from typing import Any

from hephaestus.evaluation.interface_mastery import extract_json_object

NORMALIZED_HANDOFF_VERSION = "interface-handoff.normalized.v1"
REQUIRED_RECORD_KEYS = (
    "decision",
    "action",
    "primary_variable",
    "confidence",
    "evidence_refs",
    "uncertainties",
    "rationale",
)


def normalize_handoff(*, source_role: str, raw_output: str) -> dict[str, Any]:
    parsed, parse_status = extract_json_object(raw_output)
    payload: dict[str, Any] | None = None
    raw_fallback: str | None = None
    if parsed is not None:
        payload = {key: parsed.get(key) for key in REQUIRED_RECORD_KEYS if key in parsed}
    else:
        raw_fallback = str(raw_output).strip()[:4096]

    return {
        "handoff_version": NORMALIZED_HANDOFF_VERSION,
        "source_role": str(source_role),
        "authority": "untrusted_model_output",
        "schema_copy_allowed": False,
        "parse_status": parse_status,
        "payload": payload,
        "raw_fallback": raw_fallback,
    }


def render_normalized_handoff(*, source_role: str, raw_output: str) -> str:
    """Return a stable JSON envelope suitable for paired raw-vs-normalized tests."""
    return json.dumps(
        normalize_handoff(source_role=source_role, raw_output=raw_output),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
