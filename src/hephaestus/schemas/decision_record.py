from __future__ import annotations

from dataclasses import dataclass, field

from ._base import JsonSchema


@dataclass(slots=True)
class DecisionRecord(JsonSchema):
    decision_id: str
    run_id: str
    lineage_id: str
    role: str
    action: str
    rationale: str
    confidence: float
    evidence_refs: list[str] = field(default_factory=list)
    created_at: str = ""
