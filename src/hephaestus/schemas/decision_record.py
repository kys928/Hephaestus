"""Schema: DecisionRecord."""

from __future__ import annotations

from dataclasses import dataclass

from ._base import JsonSchema


@dataclass(slots=True)
class DecisionRecord(JsonSchema):
    decision_id: str
    run_id: str
    role: str
    action: str
    rationale: str
    confidence: float
