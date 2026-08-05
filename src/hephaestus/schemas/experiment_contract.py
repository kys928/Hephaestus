"""Planner, training, and comparison contracts for controlled experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._base import JsonSchema
from .contract_common import (
    AUTONOMOUS_EXPERIMENT_CONTRACT_VERSION,
    INTERVENTION_KINDS,
    ContractIssue,
    clamp_confidence,
    normalize_vocab,
)


@dataclass(slots=True)
class InterventionProposal(JsonSchema):
    intervention_id: str
    diagnosis_report_id: str
    intervention_kind: str
    hypothesis: str
    primary_variable: str
    controlled_variables: dict[str, object] = field(default_factory=dict)
    expected_effect: str = ""
    expected_cost: dict[str, object] = field(default_factory=dict)
    required_inputs: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    success_criteria: dict[str, object] = field(default_factory=dict)
    failure_criteria: dict[str, object] = field(default_factory=dict)
    rollback_plan: str = ""
    alternatives_rejected: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)
    contract_version: str = AUTONOMOUS_EXPERIMENT_CONTRACT_VERSION

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InterventionProposal":
        values = dict(payload)
        values["intervention_kind"] = normalize_vocab(
            values.get("intervention_kind"), INTERVENTION_KINDS, "collect_more_evidence"
        )
        values["confidence"] = clamp_confidence(values.get("confidence", 0.0))
        return cls(**values)


@dataclass(slots=True)
class ExperimentProposal(JsonSchema):
    experiment_id: str
    run_id: str
    lineage_id: str
    stage_name: str
    diagnosis_report_id: str
    intervention_id: str
    primary_variable: str
    baseline_ref: str | None = None
    dataset_selection_id: str | None = None
    model_selection_id: str | None = None
    controlled_variables: dict[str, object] = field(default_factory=dict)
    training_constraints: dict[str, object] = field(default_factory=dict)
    budget: dict[str, object] = field(default_factory=dict)
    success_criteria: dict[str, object] = field(default_factory=dict)
    failure_criteria: dict[str, object] = field(default_factory=dict)
    required_evidence: list[str] = field(default_factory=list)
    required_approvals: list[str] = field(default_factory=list)
    rollback_plan: str = ""
    status: str = "pending"
    metadata: dict[str, object] = field(default_factory=dict)
    contract_version: str = AUTONOMOUS_EXPERIMENT_CONTRACT_VERSION


@dataclass(slots=True)
class TrainingRunHandle(JsonSchema):
    run_id: str
    experiment_id: str
    backend_id: str
    status: str
    checkpoint_refs: list[str] = field(default_factory=list)
    metrics_ref: str | None = None
    event_stream_ref: str | None = None
    resume_token_ref: str | None = None
    issues: list[ContractIssue] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    contract_version: str = AUTONOMOUS_EXPERIMENT_CONTRACT_VERSION

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingRunHandle":
        values = dict(payload)
        values["issues"] = [ContractIssue.from_dict(item) for item in payload.get("issues", [])]
        return cls(**values)


@dataclass(slots=True)
class TrainingControlRequest(JsonSchema):
    request_id: str
    run_id: str
    action: str
    requested_by: str
    reason: str
    approval_ref: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    contract_version: str = AUTONOMOUS_EXPERIMENT_CONTRACT_VERSION


@dataclass(slots=True)
class ExperimentComparison(JsonSchema):
    comparison_id: str
    experiment_id: str
    baseline_run_id: str | None
    candidate_run_ids: list[str] = field(default_factory=list)
    evaluation_report_refs: list[str] = field(default_factory=list)
    primary_outcome: str = "inconclusive"
    effect_summary: dict[str, object] = field(default_factory=dict)
    deterministic_gate_status: str = "unknown"
    variance_risk: str = "unknown"
    recommendation: str = "collect_more_evidence"
    evidence_refs: list[str] = field(default_factory=list)
    issues: list[ContractIssue] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)
    contract_version: str = AUTONOMOUS_EXPERIMENT_CONTRACT_VERSION

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentComparison":
        values = dict(payload)
        values["issues"] = [ContractIssue.from_dict(item) for item in payload.get("issues", [])]
        values["confidence"] = clamp_confidence(payload.get("confidence", 0.0))
        return cls(**values)
