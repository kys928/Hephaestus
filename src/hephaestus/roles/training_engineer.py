from __future__ import annotations

from dataclasses import dataclass

from hephaestus.schemas.launch_config import LaunchConfig
from hephaestus.schemas.training_plan import TrainingPlan


@dataclass(slots=True)
class TrainingEngineerRole:
    name: str = "training_engineer"

    def run(
        self,
        run_id: str,
        stage_name: str,
        artifact_root: str,
        data_contract: dict[str, object],
        backend_name: str,
        dry_run: bool,
    ) -> tuple[TrainingPlan, LaunchConfig]:
        plan = TrainingPlan(
            training_plan_id=f"train-plan-{run_id}",
            run_id=run_id,
            stage_name=stage_name,
            recipe_template="smoke_train",
            max_steps=200,
            eval_every_steps=100,
            checkpoint_every_steps=100,
            tags=["stage3", stage_name],
        )
        launch = LaunchConfig(
            launch_id=f"launch-{run_id}",
            run_id=run_id,
            backend=backend_name,
            dry_run=dry_run,
            artifact_root=artifact_root,
            parameters={
                "max_steps": "200",
                "device": "cpu",
                "processed_dataset_ref": str(data_contract["processed_dataset_ref"]),
            },
        )
        return plan, launch
