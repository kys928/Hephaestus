from __future__ import annotations

from dataclasses import dataclass

from hephaestus.planning.service import ClosedLoopExperimentPlanner
from hephaestus.schemas.diagnosis_contract import DiagnosisReport
from hephaestus.schemas.discovery_contract import DatasetSelectionDecision, ModelSelectionDecision
from hephaestus.schemas.experiment_contract import ExperimentProposal, InterventionProposal
from hephaestus.schemas.experiment_plan import ExperimentPlan


@dataclass(slots=True)
class PlannerRole:
    planner: ClosedLoopExperimentPlanner | None = None
    name: str = "planner"

    def run(self, run_id: str, stage_name: str) -> ExperimentPlan:
        """Preserve the existing bounded control-spine adapter until final wiring."""
        return ExperimentPlan(
            plan_id=f"plan-{run_id}",
            run_id=run_id,
            stage_name=stage_name,
            objective="Generate bounded dry-run outputs for all spine phases.",
            interventions=["dry_run_backend", "periodic_eval_hooks"],
            expected_outcomes=["eval_report", "judge_exit_action"],
        )

    def propose_interventions(self, diagnosis: DiagnosisReport) -> list[InterventionProposal]:
        return list(self._planner().propose_interventions(diagnosis))

    def propose_experiment(
        self,
        diagnosis: DiagnosisReport,
        intervention: InterventionProposal,
        dataset_selection: DatasetSelectionDecision | None = None,
        model_selection: ModelSelectionDecision | None = None,
    ) -> ExperimentProposal:
        return self._planner().propose_experiment(
            diagnosis,
            intervention,
            dataset_selection,
            model_selection,
        )

    def _planner(self) -> ClosedLoopExperimentPlanner:
        if self.planner is None:
            self.planner = ClosedLoopExperimentPlanner()
        return self.planner
