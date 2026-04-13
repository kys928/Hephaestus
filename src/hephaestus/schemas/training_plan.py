"""Schema: TrainingPlan."""

from __future__ import annotations

from dataclasses import dataclass

from ._base import JsonSchema


@dataclass(slots=True)
class TrainingPlan(JsonSchema):
    training_plan_id: str
    run_id: str
    recipe_template: str
    max_steps: int
