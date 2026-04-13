"""Schema: ExperimentPlan."""

from __future__ import annotations

from dataclasses import dataclass

from ._base import JsonSchema


@dataclass(slots=True)
class ExperimentPlan(JsonSchema):
    plan_id: str
    run_id: str
    intervention: str
    expected_outcome: str
