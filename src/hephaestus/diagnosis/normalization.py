"""Deterministic normalization of heterogeneous diagnostic evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

from hephaestus.schemas.contract_common import clamp_confidence

_SIGNAL_PAIRS = {
    "eval_integrity_verified": "eval_integrity_failed",
    "reproducibility_verified": "non_reproducible",
    "data_quality_verified": "data_quality_failed",
    "data_coverage_verified": "data_coverage_gap",
    "wrapper_compatible": "wrapper_mismatch",
    "tokenizer_compatible": "tokenizer_incompatible",
    "architecture_compatible": "architecture_mismatch",
    "optimizer_stable": "optimizer_pathology",
    "numerically_stable": "non_finite_loss",
    "training_sufficient": "undertraining_detected",
    "no_overfitting": "overfitting_detected",
    "decoding_verified": "decoding_mismatch",
    "runtime_healthy": "runtime_failure",
    "checkpoint_verified": "checkpoint_corrupt",
    "model_family_adequate": "model_family_limitation_detected",
}

KNOWN_SIGNALS = frozenset(
    {
        "eval_integrity_failed",
        "eval_pack_unverified",
        "deterministic_scorecard_missing",
        "evaluation_settings_mismatch",
        "non_reproducible",
        "replay_failed",
        "seed_mismatch",
        "launch_config_mismatch",
        "data_quality_failed",
        "contamination_detected",
        "malformed_data",
        "deduplication_failed",
        "data_coverage_gap",
        "domain_missing",
        "wrapper_mismatch",
        "data_format_mismatch",
        "prompt_target_boundary_mismatch",
        "tokenizer_mismatch",
        "tokenizer_incompatible",
        "special_token_mismatch",
        "architecture_mismatch",
        "checkpoint_architecture_mismatch",
        "strict_loader_contract_failed",
        "optimizer_pathology",
        "scheduler_misconfigured",
        "non_finite_loss",
        "non_finite_gradient",
        "numerical_overflow",
        "undertraining_detected",
        "training_budget_exhausted",
        "overfitting_detected",
        "train_eval_gap",
        "decoding_mismatch",
        "decoding_artifact",
        "runtime_failure",
        "hardware_interruption",
        "data_loader_failure",
        "checkpoint_hash_mismatch",
        "checkpoint_corrupt",
        "resume_checkpoint_mismatch",
        "model_family_limitation_detected",
        *_SIGNAL_PAIRS.keys(),
    }
)


@dataclass(frozen=True, slots=True)
class NormalizedEvidence:
    evidence_kind: str
    source_ref: str
    summary: str
    severity: str
    confidence: float
    signals: tuple[str, ...]
    payload: dict[str, object]
    observation_id: str


def normalize_evidence(record: Mapping[str, object], index: int) -> NormalizedEvidence:
    safe_payload = _json_safe(deepcopy(dict(record)))
    payload = safe_payload if isinstance(safe_payload, dict) else {}
    kind = (
        str(
            payload.get("evidence_kind")
            or payload.get("kind")
            or payload.get("record_kind")
            or "unknown"
        )
        .strip()
        .lower()
    )
    source_ref = str(
        payload.get("source_ref")
        or payload.get("artifact_ref")
        or payload.get("event_ref")
        or _identity_ref(payload)
        or f"inline:{index}"
    )
    signals = _extract_signals(kind, payload)
    summary = str(
        payload.get("summary") or payload.get("message") or f"{kind} evidence recorded"
    )
    severity = _severity(kind, payload, signals)
    confidence = _evidence_confidence(payload)
    canonical = json.dumps(
        {
            "kind": kind,
            "source_ref": source_ref,
            "payload": payload,
            "signals": sorted(signals),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    observation_id = f"obs-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]}"
    return NormalizedEvidence(
        evidence_kind=kind,
        source_ref=source_ref,
        summary=summary,
        severity=severity,
        confidence=confidence,
        signals=tuple(sorted(signals)),
        payload=payload,
        observation_id=observation_id,
    )


def _extract_signals(kind: str, payload: Mapping[str, object]) -> set[str]:
    signals: set[str] = set()
    explicit = payload.get("signals", payload.get("signal", []))
    if isinstance(explicit, str):
        explicit = [explicit]
    if isinstance(explicit, list):
        signals.update(str(item) for item in explicit if str(item) in KNOWN_SIGNALS)

    for key, value in _walk(payload):
        if key in KNOWN_SIGNALS and _is_true(value):
            signals.add(key)
        if key in _SIGNAL_PAIRS:
            if _is_true(value):
                signals.add(key)
            elif _is_false(value):
                signals.add(_SIGNAL_PAIRS[key])

    status = str(payload.get("status") or "").lower()
    category = str(payload.get("category") or "").lower()
    if kind in {"incident", "incident_record"}:
        signals.add("runtime_failure")
    if kind == "runtime_event" and category == "incident":
        signals.add("runtime_failure")
    if kind == "run_record":
        if status in {"failed", "crashed", "aborted", "hard_abort"}:
            signals.add("runtime_failure")
        elif status == "completed":
            signals.add("runtime_healthy")
    if kind in {"replay_verification", "replay_verification_report"}:
        if status == "reproducible":
            signals.add("reproducibility_verified")
        elif status in {"partial", "insufficient", "missing"}:
            signals.add("replay_failed")
    if kind == "eval_report":
        pack_level = str(payload.get("eval_pack_integrity_level") or "")
        score_level = str(payload.get("scorecard_integrity_level") or "")
        scorecard = payload.get("deterministic_scorecard")
        if (
            pack_level == "content_hash_verified"
            and score_level == "content_hash_verified"
            and isinstance(scorecard, dict)
            and scorecard
        ):
            signals.add("eval_integrity_verified")
        else:
            if pack_level in {"insufficient", "reference_only", "inline_unhashed", ""}:
                signals.add("eval_pack_unverified")
            if not isinstance(scorecard, dict) or not scorecard:
                signals.add("deterministic_scorecard_missing")
    if kind in {"scorecard", "deterministic_scorecard"}:
        integrity = str(
            payload.get("scorecard_integrity_level")
            or payload.get("integrity_level")
            or ""
        )
        if integrity == "content_hash_verified" and payload.get("gate_results"):
            signals.add("eval_integrity_verified")
    _add_nested_status_signals(payload, signals)
    return signals


def _add_nested_status_signals(
    payload: Mapping[str, object], signals: set[str]
) -> None:
    tokenizer = payload.get("tokenizer_compatibility")
    if isinstance(tokenizer, dict):
        status = str(tokenizer.get("status") or tokenizer.get("result") or "").lower()
        if status in {"mismatch", "incompatible", "failed"}:
            signals.add("tokenizer_mismatch")
        elif status in {"compatible", "passed", "verified"}:
            signals.add("tokenizer_compatible")
    contamination = payload.get("contamination_checks")
    if isinstance(contamination, dict) and contamination.get("detected") is True:
        signals.add("contamination_detected")
    wrapper = payload.get("wrapper_policy")
    if isinstance(wrapper, dict):
        status = str(
            wrapper.get("compatibility") or wrapper.get("status") or ""
        ).lower()
        if status in {"mismatch", "incompatible", "failed"}:
            signals.add("wrapper_mismatch")
        elif status in {"compatible", "passed", "verified"}:
            signals.add("wrapper_compatible")


def _walk(value: object) -> list[tuple[str, object]]:
    rows: list[tuple[str, object]] = []
    if isinstance(value, dict):
        for key in sorted(value):
            child = value[key]
            rows.append((str(key), child))
            rows.extend(_walk(child))
    elif isinstance(value, list):
        for child in value:
            rows.extend(_walk(child))
    return rows


def _identity_ref(payload: Mapping[str, object]) -> str:
    for key in (
        "event_id",
        "incident_id",
        "eval_id",
        "scorecard_id",
        "manifest_id",
        "report_id",
        "run_id",
        "checkpoint_ref",
        "decision_id",
        "memory_id",
    ):
        if payload.get(key):
            return f"{key}:{payload[key]}"
    return ""


def _evidence_confidence(payload: Mapping[str, object]) -> float:
    if "confidence" in payload:
        return clamp_confidence(payload.get("confidence"))
    integrity = str(
        payload.get("integrity_level")
        or payload.get("manifest_integrity_level")
        or payload.get("scorecard_integrity_level")
        or ""
    )
    if integrity in {"content_hash_verified", "complete", "reproducible"}:
        return 0.95
    if integrity == "partial":
        return 0.65
    if integrity in {"reference_only", "inline_unhashed"}:
        return 0.45
    if integrity in {"insufficient", "missing"}:
        return 0.20
    return 0.80


def _severity(kind: str, payload: Mapping[str, object], signals: set[str]) -> str:
    explicit = str(payload.get("severity") or "").lower()
    if explicit in {"info", "warning", "minor", "major", "critical"}:
        return explicit
    if {
        "runtime_failure",
        "checkpoint_corrupt",
        "non_finite_loss",
        "non_finite_gradient",
    }.intersection(signals):
        return "critical"
    if signals:
        return "major"
    return "info"


def _is_true(value: object) -> bool:
    return value is True or (
        isinstance(value, str)
        and value.lower() in {"true", "yes", "failed", "detected"}
    )


def _is_false(value: object) -> bool:
    return value is False or (
        isinstance(value, str)
        and value.lower() in {"false", "no", "passed", "verified"}
    )


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    value_type = type(value)
    return f"<unsupported:{value_type.__module__}.{value_type.__qualname__}>"
