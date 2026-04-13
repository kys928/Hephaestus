from __future__ import annotations

from dataclasses import dataclass

from hephaestus.schemas.experiment_plan import ExperimentPlan


@dataclass(slots=True)
class PlannerRole:
    name: str = "planner"

    def run(self, run_id: str, stage_name: str) -> ExperimentPlan:
        return ExperimentPlan(
            plan_id=f"plan-{run_id}",
            run_id=run_id,
            stage_name=stage_name,
            objective="Generate bounded dry-run outputs for all spine phases.",
            interventions=["dry_run_backend", "periodic_eval_hooks"],
            expected_outcomes=["eval_report", "judge_exit_action"],
        )
