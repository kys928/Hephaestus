"""Typed, JSON-safe records for bounded recovery analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hephaestus.schemas._base import JsonSchema
from hephaestus.schemas.contract_common import ContractIssue, clamp_confidence

FAILURE_CATEGORIES = (
    "transient_provider_outage",
    "transient_network_download_interruption",
    "worker_lease_loss",
    "process_crash",
    "explicit_cancellation",
    "operator_interruption",
    "out_of_memory",
    "resource_budget_exhaustion",
    "data_loader_failure",
    "malformed_or_contaminated_data",
    "tokenizer_incompatibility",
    "model_checkpoint_incompatibility",
    "checkpoint_corruption",
    "missing_checkpoint_evidence",
    "resume_token_corruption",
    "missing_metrics",
    "incomplete_evaluation",
    "deterministic_regression",
    "high_evaluation_variance",
    "replay_failure",
    "policy_or_approval_block",
    "invalid_configuration",
    "permanent_unsupported_capability",
    "poisoned_or_deprecated_lineage",
    "storage_integrity_failure",
    "state_persistence_failure",
    "unknown_inconclusive",
)

RECOVERY_ACTION_KINDS = (
    "retry_same_operation",
    "retry_after_backoff",
    "resume_verified_checkpoint",
    "restart_bounded_job",
    "request_replacement_worker",
    "reacquire_partial_artifact",
    "rerun_evaluation",
    "collect_more_evidence",
    "rollback_verified_checkpoint",
    "branch_new_experiment",
    "quarantine_lineage",
    "stop",
    "escalate_for_human_approval",
)


@dataclass(slots=True)
class RecoveryRequest(JsonSchema):
    request_id: str
    run_id: str
    experiment_id: str
    lineage_id: str
    stage_name: str
    operation_id: str
    evidence: list[dict[str, object]] = field(default_factory=list)
    approval_evidence: list[dict[str, object]] = field(default_factory=list)
    constraints: dict[str, object] = field(default_factory=dict)
    requested_action_kind: str | None = None
    requested_by: str = "incident_responder"


@dataclass(slots=True)
class NormalizedFailureEvidence(JsonSchema):
    evidence_id: str
    evidence_kind: str
    source_ref: str
    summary: str
    confidence: float
    signals: list[str] = field(default_factory=list)
    payload: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NormalizedFailureEvidence:
        values = dict(payload)
        values["confidence"] = clamp_confidence(values.get("confidence", 0.0))
        values["signals"] = [str(item) for item in values.get("signals", [])]
        values["payload"] = dict(values.get("payload", {}))
        return cls(**values)


@dataclass(slots=True)
class FailureClassification(JsonSchema):
    classification_id: str
    category: str
    likely_failure_domain: str
    confidence: float
    retryability: str
    evidence_refs: list[str] = field(default_factory=list)
    contradicting_evidence_refs: list[str] = field(default_factory=list)
    alternative_categories: list[str] = field(default_factory=list)
    requires_new_evidence: bool = False
    requires_approval: bool = False
    safe_to_automate: bool = False
    failure_signature: str = ""
    evidence_fingerprint: str = ""
    issues: list[ContractIssue] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FailureClassification:
        values = dict(payload)
        values["confidence"] = clamp_confidence(values.get("confidence", 0.0))
        values["issues"] = [
            ContractIssue.from_dict(item) for item in values.get("issues", [])
        ]
        return cls(**values)


@dataclass(slots=True)
class RetryBudgetDecision(JsonSchema):
    allowed: bool
    counts: dict[str, int] = field(default_factory=dict)
    limits: dict[str, int] = field(default_factory=dict)
    exhausted_scopes: list[str] = field(default_factory=list)
    identical_evidence_attempts: int = 0
    cumulative_cost: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BackoffDecision(JsonSchema):
    required: bool
    delay_seconds: int
    policy_kind: str
    attempt_index: int
    deterministic_jitter_seconds: int = 0
    maximum_delay_seconds: int = 0
    reset_after_progress: bool = True


@dataclass(slots=True)
class CheckpointRecoveryDecision(JsonSchema):
    allowed: bool
    checkpoint_ref: str | None = None
    resume_token_ref: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    issues: list[ContractIssue] = field(default_factory=list)
    compatibility: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CheckpointRecoveryDecision:
        values = dict(payload)
        values["issues"] = [
            ContractIssue.from_dict(item) for item in values.get("issues", [])
        ]
        return cls(**values)


@dataclass(slots=True)
class RecoveryRecommendation(JsonSchema):
    recommendation_id: str
    action_kind: str
    registry_action: str | None
    rationale: str
    executable: bool
    reversible: bool
    approval_required: bool
    evidence_refs: list[str] = field(default_factory=list)
    parameters: dict[str, object] = field(default_factory=dict)
    approval_ref: str | None = None


@dataclass(slots=True)
class RecoveryDecision(JsonSchema):
    decision_id: str
    request_id: str
    run_id: str
    experiment_id: str
    lineage_id: str
    stage_name: str
    status: str
    classification: FailureClassification
    recommendation: RecoveryRecommendation
    budget: RetryBudgetDecision
    backoff: BackoffDecision
    checkpoint: CheckpointRecoveryDecision | None
    attempt_id: str
    evidence_refs: list[str] = field(default_factory=list)
    issues: list[ContractIssue] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RecoveryDecision:
        values = dict(payload)
        values["classification"] = FailureClassification.from_dict(
            values["classification"]
        )
        values["recommendation"] = RecoveryRecommendation.from_dict(
            values["recommendation"]
        )
        values["budget"] = RetryBudgetDecision.from_dict(values["budget"])
        values["backoff"] = BackoffDecision.from_dict(values["backoff"])
        if values.get("checkpoint") is not None:
            values["checkpoint"] = CheckpointRecoveryDecision.from_dict(
                values["checkpoint"]
            )
        values["issues"] = [
            ContractIssue.from_dict(item) for item in values.get("issues", [])
        ]
        return cls(**values)


@dataclass(slots=True)
class RecoveryAttempt(JsonSchema):
    attempt_id: str
    request_id: str
    run_id: str
    experiment_id: str
    lineage_id: str
    operation_id: str
    failure_signature: str
    evidence_fingerprint: str
    action_kind: str
    registry_action: str | None
    status: str
    input_evidence_refs: list[str] = field(default_factory=list)
    result_ref: str | None = None
    cost: float = 0.0
    progress_observed: bool = False
    prior_state: str | None = None
    next_state: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class RecoveryExecutionResult(JsonSchema):
    attempt_id: str
    status: str
    action_kind: str
    result_ref: str | None = None
    issues: list[ContractIssue] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
