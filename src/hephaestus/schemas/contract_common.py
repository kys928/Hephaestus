"""Shared autonomous-experiment contract vocabulary.

This module defines transport-safe issue records and finite vocabularies. It does
not perform policy decisions or execute subsystem behavior.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._base import JsonSchema

AUTONOMOUS_EXPERIMENT_CONTRACT_VERSION = "autonomous-experiment.v1"

CONTRACT_STATUSES = (
    "pending", "ready", "selected", "inconclusive", "blocked", "failed", "completed",
)
ISSUE_CATEGORIES = (
    "invalid_request", "unsupported_capability", "policy_blocked", "missing_evidence",
    "provider_unavailable", "candidate_not_found", "incompatible_candidate",
    "license_unknown", "provenance_unknown", "contamination_risk", "artifact_integrity",
    "runtime_failure", "evaluation_inconclusive", "budget_exceeded", "approval_required",
    "internal_contract_violation",
)
FAILURE_DOMAINS = (
    "evaluation_integrity", "launch_or_reproducibility", "data_quality", "data_coverage",
    "data_format_or_wrapper", "tokenizer", "architecture", "optimizer_or_scheduler",
    "numerical_instability", "undertraining", "overfitting", "decoding",
    "runtime_or_system", "checkpoint_integrity", "model_family_limitation", "inconclusive",
)
INTERVENTION_KINDS = (
    "collect_more_evidence", "repair_evaluation", "repair_data", "replace_or_mix_dataset",
    "change_preprocessing", "change_tokenizer", "change_training_recipe", "resume_training",
    "change_model", "rollback", "branch", "restart", "stop",
)
SELECTION_STATUSES = ("selected", "inconclusive", "blocked")


def clamp_confidence(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def normalize_vocab(value: object, allowed: tuple[str, ...], default: str) -> str:
    candidate = str(value).strip() if value is not None else ""
    return candidate if candidate in allowed else default


@dataclass(slots=True)
class ContractIssue(JsonSchema):
    code: str
    category: str
    message: str
    retryable: bool = False
    blocking: bool = False
    evidence_refs: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    contract_version: str = AUTONOMOUS_EXPERIMENT_CONTRACT_VERSION

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContractIssue":
        return cls(
            code=str(payload.get("code", "contract_issue")),
            category=normalize_vocab(payload.get("category"), ISSUE_CATEGORIES, "internal_contract_violation"),
            message=str(payload.get("message", "")),
            retryable=bool(payload.get("retryable", False)),
            blocking=bool(payload.get("blocking", False)),
            evidence_refs=[str(item) for item in payload.get("evidence_refs", [])],
            metadata=dict(payload.get("metadata", {})),
            contract_version=str(payload.get("contract_version", AUTONOMOUS_EXPERIMENT_CONTRACT_VERSION)),
        )
