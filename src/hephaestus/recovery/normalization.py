"""Deterministic normalization for heterogeneous recovery evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy

from hephaestus.recovery.models import NormalizedFailureEvidence
from hephaestus.schemas.contract_common import clamp_confidence

KNOWN_SIGNALS = frozenset(
    {
        "provider_unavailable",
        "provider_healthy",
        "transient_provider_outage",
        "network_interruption",
        "download_interrupted",
        "lease_expired",
        "heartbeat_missing",
        "worker_process_missing",
        "process_alive_unattached",
        "duplicate_ownership",
        "late_completion",
        "stale_result",
        "exclusive_ownership_verified",
        "process_crash",
        "explicit_cancellation",
        "operator_interruption",
        "out_of_memory",
        "budget_exhausted",
        "data_loader_failure",
        "malformed_data",
        "contamination_detected",
        "tokenizer_incompatible",
        "model_checkpoint_incompatible",
        "checkpoint_corrupt",
        "checkpoint_hash_mismatch",
        "checkpoint_verified",
        "checkpoint_missing",
        "resume_token_corrupt",
        "resume_token_valid",
        "metrics_missing",
        "evaluation_incomplete",
        "evaluation_complete",
        "deterministic_regression",
        "high_evaluation_variance",
        "evaluation_variance_bounded",
        "replay_failed",
        "replay_verified",
        "policy_blocked",
        "approval_required",
        "approval_verified",
        "invalid_configuration",
        "configuration_verified",
        "unsupported_capability",
        "lineage_poisoned",
        "lineage_deprecated",
        "lineage_archived",
        "lineage_trusted",
        "storage_integrity_failure",
        "state_persistence_failure",
        "artifact_partial",
        "genuine_progress",
    }
)

_CODE_SIGNALS = {
    "provider_unavailable": "provider_unavailable",
    "network_interruption": "network_interruption",
    "download_interrupted": "download_interrupted",
    "lease_expired": "lease_expired",
    "heartbeat_missing": "heartbeat_missing",
    "worker_process_missing": "worker_process_missing",
    "duplicate_ownership": "duplicate_ownership",
    "stale_result": "stale_result",
    "non_zero_exit": "process_crash",
    "process_launch_failure": "process_crash",
    "resume_launch_failure": "process_crash",
    "out_of_memory": "out_of_memory",
    "budget_exceeded": "budget_exhausted",
    "data_loader_failure": "data_loader_failure",
    "malformed_data": "malformed_data",
    "contamination_detected": "contamination_detected",
    "tokenizer_mismatch": "tokenizer_incompatible",
    "tokenizer_incompatible": "tokenizer_incompatible",
    "resume_compatibility_mismatch": "model_checkpoint_incompatible",
    "resume_request_mismatch": "model_checkpoint_incompatible",
    "checkpoint_hash_mismatch": "checkpoint_hash_mismatch",
    "checkpoint_integrity_failure": "checkpoint_corrupt",
    "checkpoint_corrupt": "checkpoint_corrupt",
    "checkpoint_integrity_missing": "checkpoint_missing",
    "resume_token_missing": "checkpoint_missing",
    "resume_evidence_invalid": "resume_token_corrupt",
    "missing_required_artifact": "artifact_partial",
    "missing_metrics": "metrics_missing",
    "required_samples_missing": "evaluation_incomplete",
    "replay_failed": "replay_failed",
    "approval_required": "approval_required",
    "policy_blocked": "policy_blocked",
    "invalid_configuration": "invalid_configuration",
    "unsupported_backend": "unsupported_capability",
    "unsupported_capability": "unsupported_capability",
    "storage_integrity_failure": "storage_integrity_failure",
    "state_persistence_failure": "state_persistence_failure",
}


def normalize_failure_evidence(
    records: list[Mapping[str, object]],
) -> list[NormalizedFailureEvidence]:
    normalized: dict[str, NormalizedFailureEvidence] = {}
    for index, record in enumerate(records):
        item = _normalize_one(record, index)
        normalized[item.evidence_id] = item
    return [normalized[key] for key in sorted(normalized)]


def _normalize_one(
    record: Mapping[str, object], index: int
) -> NormalizedFailureEvidence:
    safe = _json_safe(deepcopy(dict(record)))
    payload = safe if isinstance(safe, dict) else {}
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
    confidence = _confidence(payload)
    canonical = json.dumps(
        {
            "kind": kind,
            "source_ref": source_ref,
            "signals": sorted(signals),
            "payload": payload,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    evidence_id = f"re-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]}"
    return NormalizedFailureEvidence(
        evidence_id=evidence_id,
        evidence_kind=kind,
        source_ref=source_ref,
        summary=str(
            payload.get("summary") or payload.get("message") or f"{kind} evidence"
        ),
        confidence=confidence,
        signals=sorted(signals),
        payload=payload,
    )


def _extract_signals(kind: str, payload: Mapping[str, object]) -> set[str]:
    signals: set[str] = set()
    explicit = payload.get("signals", payload.get("signal", []))
    if isinstance(explicit, str):
        explicit = [explicit]
    if isinstance(explicit, list):
        signals.update(str(item) for item in explicit if str(item) in KNOWN_SIGNALS)

    for key, value in _walk(payload):
        normalized_key = str(key).strip().lower()
        if normalized_key == "signals" and isinstance(value, list):
            signals.update(str(item) for item in value if str(item) in KNOWN_SIGNALS)
        if normalized_key in KNOWN_SIGNALS and _explicit_true(value):
            signals.add(normalized_key)
        mapped = _CODE_SIGNALS.get(str(value).strip().lower())
        if (
            normalized_key in {"code", "category", "error_ref", "reason", "outcome"}
            and mapped
        ):
            signals.add(mapped)

    status = str(payload.get("status") or "").strip().lower()
    category = str(payload.get("category") or "").strip().lower()
    if kind in {"job", "job_record"}:
        if (
            status == "expired"
            or str(payload.get("error_ref") or "") == "lease_expired"
        ):
            signals.add("lease_expired")
        elif status == "failed" and payload.get("error_ref"):
            signals.add("process_crash")
        elif status == "cancelled":
            signals.add("explicit_cancellation")
    if kind in {"training_handle", "training_run_handle", "run_record"}:
        if status == "cancelled":
            signals.add("explicit_cancellation")
        elif status == "interrupted" and payload.get("operator_interruption") is True:
            signals.add("operator_interruption")
    if kind in {"checkpoint", "checkpoint_record"}:
        integrity = str(payload.get("integrity_level") or "").lower()
        if integrity == "content_hash_verified" and payload.get("content_hash"):
            signals.add("checkpoint_verified")
        if payload.get("exists") is False:
            signals.add("checkpoint_missing")
        if payload.get("hash_verified") is False:
            signals.add("checkpoint_hash_mismatch")
    if kind in {"resume_token", "resume_token_record"}:
        if payload.get("valid") is True and payload.get("exists", True) is True:
            signals.add("resume_token_valid")
        elif payload.get("valid") is False:
            signals.add("resume_token_corrupt")
    if kind in {"experiment_comparison", "evaluation_comparison", "eval_report"}:
        outcome = str(payload.get("primary_outcome") or "").lower()
        gate = str(payload.get("deterministic_gate_status") or "").lower()
        variance = str(payload.get("variance_risk") or "").lower()
        if outcome == "regressed" and gate == "failed":
            signals.add("deterministic_regression")
        if outcome in {"invalid_comparison", "inconclusive"} or gate in {
            "incomplete",
            "incompatible",
        }:
            signals.add("evaluation_incomplete")
        if variance == "high":
            signals.add("high_evaluation_variance")
        elif variance in {"low", "moderate"}:
            signals.add("evaluation_variance_bounded")
    if kind in {"replay_verification", "replay_verification_report"}:
        if status == "reproducible":
            signals.add("replay_verified")
        elif status in {"partial", "insufficient", "missing"}:
            signals.add("replay_failed")
    if kind in {"lineage", "lineage_state"}:
        if status in {"poisoned", "deprecated", "archived"}:
            signals.add(f"lineage_{status}")
        elif status in {"promising", "stable"}:
            signals.add("lineage_trusted")
    if kind in {"incident", "incident_record", "runtime_event"} and category in {
        "operator_interruption",
        "explicit_cancellation",
    }:
        signals.add(category)
    return signals


def evidence_fingerprint(evidence: list[NormalizedFailureEvidence]) -> str:
    payload = [
        {
            "id": item.evidence_id,
            "ref": item.source_ref,
            "signals": item.signals,
        }
        for item in evidence
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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


def _explicit_true(value: object) -> bool:
    return value is True or (
        isinstance(value, str)
        and value.strip().lower() in {"true", "yes", "detected", "failed"}
    )


def _confidence(payload: Mapping[str, object]) -> float:
    if "confidence" in payload:
        return clamp_confidence(payload.get("confidence"))
    integrity = str(payload.get("integrity_level") or "").lower()
    if integrity in {"content_hash_verified", "reproducible", "complete"}:
        return 0.95
    if integrity == "partial":
        return 0.65
    if integrity in {"insufficient", "missing"}:
        return 0.2
    return 0.8


def _identity_ref(payload: Mapping[str, object]) -> str:
    for key in (
        "report_id",
        "incident_id",
        "event_id",
        "run_id",
        "checkpoint_ref",
        "job_id",
        "memory_id",
        "attempt_id",
    ):
        if payload.get(key):
            return f"{key}:{payload[key]}"
    return ""


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return f"<unsupported:{type(value).__module__}.{type(value).__qualname__}>"
