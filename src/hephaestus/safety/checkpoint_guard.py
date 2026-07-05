"""Checkpoint candidate and promotion-boundary safety checks."""

from __future__ import annotations

from hephaestus.schemas.safety_guard import SafetyGuardInput, SafetyGuardResult
from hephaestus.safety._helpers import mapping, result, sequence_of_mappings, text


def check_checkpoint_candidates(inp: SafetyGuardInput) -> SafetyGuardResult:
    candidates = sequence_of_mappings(inp.payload.get("checkpoint_candidates"))
    reasons: list[str] = []
    warnings: list[str] = []
    if not candidates:
        reasons.append("checkpoint_candidates_missing")
    for index, candidate in enumerate(candidates):
        if not text(candidate.get("checkpoint_ref")):
            reasons.append(f"checkpoint_ref_missing:{index}")
        if not text(candidate.get("content_hash")):
            warnings.append(f"checkpoint_content_hash_missing:{index}")
    return result(inp, reasons, warnings, {"candidate_count": len(candidates)})


def check_selected_checkpoint(inp: SafetyGuardInput) -> SafetyGuardResult:
    resolution = mapping(inp.payload.get("checkpoint_resolution")) or inp.payload
    selected = text(resolution.get("selected_checkpoint_ref") or resolution.get("checkpoint_ref"))
    reasons = [] if selected else ["selected_checkpoint_ref_missing"]
    warnings = [] if text(resolution.get("content_hash")) else ["selected_checkpoint_content_hash_missing"]
    return result(inp, reasons, warnings, {"selected_checkpoint_ref": selected or None})
