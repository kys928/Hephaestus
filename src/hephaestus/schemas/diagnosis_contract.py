"""Schemas exchanged by evidence-based diagnosis components."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._base import JsonSchema
from .contract_common import (
    AUTONOMOUS_EXPERIMENT_CONTRACT_VERSION,
    FAILURE_DOMAINS,
    INTERVENTION_KINDS,
    ContractIssue,
    clamp_confidence,
    normalize_vocab,
)


@dataclass(slots=True)
class DiagnosisRequest(JsonSchema):
    request_id: str
    run_id: str
    lineage_id: str
    stage_name: str
    observed_failures: list[dict[str, object]] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    constraints: dict[str, object] = field(default_factory=dict)
    requested_by: str = "judge_entry"
    contract_version: str = AUTONOMOUS_EXPERIMENT_CONTRACT_VERSION


@dataclass(slots=True)
class EvidenceObservation(JsonSchema):
    observation_id: str
    evidence_kind: str
    source_ref: str
    summary: str
    severity: str = "info"
    confidence: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvidenceObservation":
        return cls(
            observation_id=str(payload.get("observation_id", "")),
            evidence_kind=str(payload.get("evidence_kind", "unknown")),
            source_ref=str(payload.get("source_ref", "")),
            summary=str(payload.get("summary", "")),
            severity=str(payload.get("severity", "info")),
            confidence=clamp_confidence(payload.get("confidence", 0.0)),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class DiagnosticHypothesis(JsonSchema):
    hypothesis_id: str
    failure_domain: str
    summary: str
    supporting_evidence_refs: list[str] = field(default_factory=list)
    contradicting_evidence_refs: list[str] = field(default_factory=list)
    required_tests: list[str] = field(default_factory=list)
    recommended_intervention_kinds: list[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DiagnosticHypothesis":
        interventions = [
            normalize_vocab(item, INTERVENTION_KINDS, "collect_more_evidence")
            for item in payload.get("recommended_intervention_kinds", [])
        ]
        return cls(
            hypothesis_id=str(payload.get("hypothesis_id", "")),
            failure_domain=normalize_vocab(payload.get("failure_domain"), FAILURE_DOMAINS, "inconclusive"),
            summary=str(payload.get("summary", "")),
            supporting_evidence_refs=[str(item) for item in payload.get("supporting_evidence_refs", [])],
            contradicting_evidence_refs=[str(item) for item in payload.get("contradicting_evidence_refs", [])],
            required_tests=[str(item) for item in payload.get("required_tests", [])],
            recommended_intervention_kinds=interventions,
            confidence=clamp_confidence(payload.get("confidence", 0.0)),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class DiagnosisReport(JsonSchema):
    report_id: str
    request_id: str
    run_id: str
    lineage_id: str
    stage_name: str
    status: str = "inconclusive"
    observations: list[EvidenceObservation] = field(default_factory=list)
    hypotheses: list[DiagnosticHypothesis] = field(default_factory=list)
    leading_hypothesis_id: str | None = None
    missing_evidence: list[str] = field(default_factory=list)
    issues: list[ContractIssue] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)
    contract_version: str = AUTONOMOUS_EXPERIMENT_CONTRACT_VERSION

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DiagnosisReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            request_id=str(payload.get("request_id", "")),
            run_id=str(payload.get("run_id", "")),
            lineage_id=str(payload.get("lineage_id", "")),
            stage_name=str(payload.get("stage_name", "")),
            status=str(payload.get("status", "inconclusive")),
            observations=[EvidenceObservation.from_dict(item) for item in payload.get("observations", [])],
            hypotheses=[DiagnosticHypothesis.from_dict(item) for item in payload.get("hypotheses", [])],
            leading_hypothesis_id=payload.get("leading_hypothesis_id"),
            missing_evidence=[str(item) for item in payload.get("missing_evidence", [])],
            issues=[ContractIssue.from_dict(item) for item in payload.get("issues", [])],
            confidence=clamp_confidence(payload.get("confidence", 0.0)),
            metadata=dict(payload.get("metadata", {})),
            contract_version=str(payload.get("contract_version", AUTONOMOUS_EXPERIMENT_CONTRACT_VERSION)),
        )
