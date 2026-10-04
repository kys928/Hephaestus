"""Typed records for Phase II model-to-model interface evaluation.

Phase II does not grant model outputs authority.  It records the producer handoff,
consumer interpretation, deterministic facts, and an explicit scorecard so that
interface failures remain auditable and cannot silently become executable state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._base import JsonSchema

INTERFACE_MASTERY_CONTRACT_VERSION = "interface-mastery.v1"
INTERFACE_IDS = (
    "diagnosis_to_planner",
    "planner_to_judge",
    "evaluator_to_judge",
    "judge_to_controller",
)


@dataclass(slots=True)
class InterfaceHandoff(JsonSchema):
    case_id: str
    interface_id: str
    producer_role: str
    consumer_role: str
    producer_output: str
    verified_facts: dict[str, object] = field(default_factory=dict)
    allowed_evidence_refs: list[str] = field(default_factory=list)
    contract_version: str = INTERFACE_MASTERY_CONTRACT_VERSION

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InterfaceHandoff":
        interface_id = str(payload.get("interface_id", ""))
        if interface_id not in INTERFACE_IDS:
            raise ValueError(f"unknown Phase II interface: {interface_id}")
        return cls(
            case_id=str(payload.get("case_id", "")),
            interface_id=interface_id,
            producer_role=str(payload.get("producer_role", "")),
            consumer_role=str(payload.get("consumer_role", "")),
            producer_output=str(payload.get("producer_output", "")),
            verified_facts=dict(payload.get("verified_facts", {})),
            allowed_evidence_refs=[str(x) for x in payload.get("allowed_evidence_refs", [])],
            contract_version=str(
                payload.get("contract_version", INTERFACE_MASTERY_CONTRACT_VERSION)
            ),
        )


@dataclass(slots=True)
class InterfaceScorecard(JsonSchema):
    case_id: str
    interface_id: str
    producer_schema_valid: bool
    consumer_schema_valid: bool
    producer_contract_exact: bool
    consumer_contract_exact: bool
    evidence_grounded: bool
    interface_invariant_passed: bool
    deterministic_boundary_passed: bool
    quality_100: float
    hallucinated_evidence_refs: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    contract_version: str = INTERFACE_MASTERY_CONTRACT_VERSION


@dataclass(slots=True)
class InterfaceMasterySummary(JsonSchema):
    protocol_id: str
    sample_count: int
    overall_quality_100: float
    schema_compliance: float
    evidence_grounding: float
    interface_invariant_pass_rate: float
    deterministic_boundary_pass_rate: float
    consumer_exact_contract_pass_rate: float
    hallucination_rate: float
    per_interface: dict[str, dict[str, float]] = field(default_factory=dict)
    certification_checks: dict[str, bool] = field(default_factory=dict)
    certified: bool = False
    disposition: str = "inconclusive"
    contract_version: str = INTERFACE_MASTERY_CONTRACT_VERSION
