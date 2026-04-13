from __future__ import annotations

from dataclasses import dataclass

from hephaestus.schemas.launch_config import LaunchConfig
from hephaestus.schemas.training_plan import TrainingPlan


@dataclass(slots=True)
class TrainingEngineerRole:
    name: str = "training_engineer"

    def run(self, run_id: str, stage_name: str, artifact_root: str) -> tuple[TrainingPlan, LaunchConfig]:
        plan = TrainingPlan(
            training_plan_id=f"train-plan-{run_id}",
            run_id=run_id,
            stage_name=stage_name,
            recipe_template="smoke_train",
            max_steps=200,
            eval_every_steps=100,
            checkpoint_every_steps=100,
            tags=["dry_run", stage_name],
        )
        launch = LaunchConfig(
            launch_id=f"launch-{run_id}",
            run_id=run_id,
            backend="dry_run",
            dry_run=True,
            artifact_root=artifact_root,
            parameters={"max_steps": "200", "device": "cpu"},
        )
        return plan, launch
