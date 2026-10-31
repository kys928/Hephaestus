"""Admitted-domain request derived from the frozen finite model vocabulary."""
from dataclasses import dataclass, field
from ._base import JsonSchema


@dataclass(slots=True)
class ModelInterventionSelection(JsonSchema):
    intervention_kind: str
    primary_variable: str
    confidence: float
    evidence_refs: list[str]
    inference_receipt_ref: str
    rationale: str
    model_decision: dict[str, object] = field(default_factory=dict)
