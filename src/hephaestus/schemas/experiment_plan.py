from __future__ import annotations

from dataclasses import dataclass, field

from ._base import JsonSchema


@dataclass(slots=True)
class ExperimentPlan(JsonSchema):
    plan_id: str
    run_id: str
    stage_name: str
    objective: str
    interventions: list[str] = field(default_factory=list)
    expected_outcomes: list[str] = field(default_factory=list)
