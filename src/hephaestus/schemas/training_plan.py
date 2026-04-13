from __future__ import annotations

from dataclasses import dataclass, field

from ._base import JsonSchema


@dataclass(slots=True)
class TrainingPlan(JsonSchema):
    training_plan_id: str
    run_id: str
    stage_name: str
    recipe_template: str
    max_steps: int
    eval_every_steps: int
    checkpoint_every_steps: int
    tags: list[str] = field(default_factory=list)
