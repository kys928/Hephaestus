"""Governed dataset and model discovery/selection contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._base import JsonSchema
from .contract_common import (
    AUTONOMOUS_EXPERIMENT_CONTRACT_VERSION,
    SELECTION_STATUSES,
    ContractIssue,
    clamp_confidence,
    normalize_vocab,
)


@dataclass(slots=True)
class DatasetSearchRequest(JsonSchema):
    request_id: str
    diagnosis_report_id: str
    problem_statement: str
    capability_targets: list[str] = field(default_factory=list)
    required_languages: list[str] = field(default_factory=list)
    required_domains: list[str] = field(default_factory=list)
    required_formats: list[str] = field(default_factory=list)
    tokenizer_ref: str | None = None
    model_constraints: dict[str, object] = field(default_factory=dict)
    size_constraints: dict[str, object] = field(default_factory=dict)
    license_allowlist: list[str] = field(default_factory=list)
    license_denylist: list[str] = field(default_factory=list)
    provider_allowlist: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    contract_version: str = AUTONOMOUS_EXPERIMENT_CONTRACT_VERSION


@dataclass(slots=True)
class DatasetCandidate(JsonSchema):
    candidate_id: str
    provider_id: str
    dataset_id: str
    revision: str | None = None
    splits: list[str] = field(default_factory=list)
    task_types: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    format_profile: dict[str, object] = field(default_factory=dict)
    estimated_rows: int | None = None
    estimated_bytes: int | None = None
    license: str | None = None
    provenance: dict[str, object] = field(default_factory=dict)
    trust_level: str = "unknown"
    compatibility: dict[str, object] = field(default_factory=dict)
    risk_signals: list[str] = field(default_factory=list)
    artifact_ref: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    missing_metadata: list[str] = field(default_factory=list)
    score_components: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    contract_version: str = AUTONOMOUS_EXPERIMENT_CONTRACT_VERSION


@dataclass(slots=True)
class DatasetSelectionDecision(JsonSchema):
    decision_id: str
    request_id: str
    status: str
    selected_candidate_ids: list[str] = field(default_factory=list)
    ranked_candidate_ids: list[str] = field(default_factory=list)
    rejected_candidates: dict[str, str] = field(default_factory=dict)
    selection_rationale: str = ""
    mixture_weights: dict[str, float] = field(default_factory=dict)
    preprocessing_requirements: dict[str, object] = field(default_factory=dict)
    required_approvals: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    issues: list[ContractIssue] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)
    contract_version: str = AUTONOMOUS_EXPERIMENT_CONTRACT_VERSION

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetSelectionDecision":
        return cls(
            decision_id=str(payload.get("decision_id", "")),
            request_id=str(payload.get("request_id", "")),
            status=normalize_vocab(payload.get("status"), SELECTION_STATUSES, "inconclusive"),
            selected_candidate_ids=[str(item) for item in payload.get("selected_candidate_ids", [])],
            ranked_candidate_ids=[str(item) for item in payload.get("ranked_candidate_ids", [])],
            rejected_candidates={str(k): str(v) for k, v in payload.get("rejected_candidates", {}).items()},
            selection_rationale=str(payload.get("selection_rationale", "")),
            mixture_weights={str(k): float(v) for k, v in payload.get("mixture_weights", {}).items()},
            preprocessing_requirements=dict(payload.get("preprocessing_requirements", {})),
            required_approvals=[str(item) for item in payload.get("required_approvals", [])],
            evidence_refs=[str(item) for item in payload.get("evidence_refs", [])],
            issues=[ContractIssue.from_dict(item) for item in payload.get("issues", [])],
            confidence=clamp_confidence(payload.get("confidence", 0.0)),
            metadata=dict(payload.get("metadata", {})),
            contract_version=str(payload.get("contract_version", AUTONOMOUS_EXPERIMENT_CONTRACT_VERSION)),
        )


@dataclass(slots=True)
class ModelSearchRequest(JsonSchema):
    request_id: str
    diagnosis_report_id: str
    problem_statement: str
    task_requirements: list[str] = field(default_factory=list)
    architecture_constraints: dict[str, object] = field(default_factory=dict)
    tokenizer_constraints: dict[str, object] = field(default_factory=dict)
    runtime_constraints: dict[str, object] = field(default_factory=dict)
    budget_constraints: dict[str, object] = field(default_factory=dict)
    license_allowlist: list[str] = field(default_factory=list)
    provider_allowlist: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    contract_version: str = AUTONOMOUS_EXPERIMENT_CONTRACT_VERSION


@dataclass(slots=True)
class ModelCandidate(JsonSchema):
    candidate_id: str
    provider_id: str
    model_id: str
    revision: str | None = None
    architecture_family: str | None = None
    parameter_count: int | None = None
    context_length: int | None = None
    tokenizer_ref: str | None = None
    license: str | None = None
    capabilities: list[str] = field(default_factory=list)
    runtime_requirements: dict[str, object] = field(default_factory=dict)
    compatibility: dict[str, object] = field(default_factory=dict)
    risk_signals: list[str] = field(default_factory=list)
    artifact_ref: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    missing_metadata: list[str] = field(default_factory=list)
    score_components: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    contract_version: str = AUTONOMOUS_EXPERIMENT_CONTRACT_VERSION


@dataclass(slots=True)
class ModelSelectionDecision(JsonSchema):
    decision_id: str
    request_id: str
    status: str
    selected_candidate_id: str | None = None
    ranked_candidate_ids: list[str] = field(default_factory=list)
    rejected_candidates: dict[str, str] = field(default_factory=dict)
    selection_rationale: str = ""
    required_approvals: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    issues: list[ContractIssue] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)
    contract_version: str = AUTONOMOUS_EXPERIMENT_CONTRACT_VERSION

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelSelectionDecision":
        return cls(
            decision_id=str(payload.get("decision_id", "")),
            request_id=str(payload.get("request_id", "")),
            status=normalize_vocab(payload.get("status"), SELECTION_STATUSES, "inconclusive"),
            selected_candidate_id=payload.get("selected_candidate_id"),
            ranked_candidate_ids=[str(item) for item in payload.get("ranked_candidate_ids", [])],
            rejected_candidates={str(k): str(v) for k, v in payload.get("rejected_candidates", {}).items()},
            selection_rationale=str(payload.get("selection_rationale", "")),
            required_approvals=[str(item) for item in payload.get("required_approvals", [])],
            evidence_refs=[str(item) for item in payload.get("evidence_refs", [])],
            issues=[ContractIssue.from_dict(item) for item in payload.get("issues", [])],
            confidence=clamp_confidence(payload.get("confidence", 0.0)),
            metadata=dict(payload.get("metadata", {})),
            contract_version=str(payload.get("contract_version", AUTONOMOUS_EXPERIMENT_CONTRACT_VERSION)),
        )
