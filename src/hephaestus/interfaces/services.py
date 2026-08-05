"""Subsystem service protocols used by the final integration branch."""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from hephaestus.schemas.diagnosis_contract import DiagnosisReport, DiagnosisRequest
from hephaestus.schemas.discovery_contract import DatasetSelectionDecision, ModelSelectionDecision
from hephaestus.schemas.experiment_contract import (
    ExperimentComparison,
    ExperimentProposal,
    InterventionProposal,
    TrainingControlRequest,
    TrainingRunHandle,
)


@runtime_checkable
class DiagnosisService(Protocol):
    def diagnose(self, request: DiagnosisRequest) -> DiagnosisReport: ...


@runtime_checkable
class ExperimentPlanningService(Protocol):
    def propose_interventions(self, diagnosis: DiagnosisReport) -> Sequence[InterventionProposal]: ...
    def propose_experiment(
        self,
        diagnosis: DiagnosisReport,
        intervention: InterventionProposal,
        dataset_selection: DatasetSelectionDecision | None,
        model_selection: ModelSelectionDecision | None,
    ) -> ExperimentProposal: ...


@runtime_checkable
class TrainingLifecycleService(Protocol):
    def launch(self, proposal: ExperimentProposal) -> TrainingRunHandle: ...
    def control(self, request: TrainingControlRequest) -> TrainingRunHandle: ...
    def status(self, run_id: str) -> TrainingRunHandle: ...


@runtime_checkable
class ExperimentEvaluationService(Protocol):
    def compare(self, proposal: ExperimentProposal, runs: Sequence[TrainingRunHandle]) -> ExperimentComparison: ...
