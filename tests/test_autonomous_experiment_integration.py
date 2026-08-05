from __future__ import annotations

from dataclasses import dataclass, field

from hephaestus.control.autonomous_experiment import (
    ApprovalAwareDatasetSelectionService,
    AutonomousExperimentCoordinator,
    GuardedTrainingLifecycleService,
    InMemoryIntegrationRecordSink,
    normalize_diagnostic_truth_values,
)
from hephaestus.data.registry import DatasetProviderRegistry
from hephaestus.planning import ClosedLoopExperimentPlanner
from hephaestus.providers.models import DeterministicModelSelectionService
from hephaestus.schemas.diagnosis_contract import DiagnosisReport
from hephaestus.schemas.discovery_contract import (
    DatasetCandidate,
    DatasetSearchRequest,
    ModelCandidate,
    ModelSearchRequest,
)
from hephaestus.schemas.experiment_contract import (
    ExperimentComparison,
    ExperimentProposal,
    TrainingControlRequest,
    TrainingRunHandle,
)


def test_key_aware_diagnostic_truth_normalization() -> None:
    normalized = normalize_diagnostic_truth_values(
        {
            "eval_integrity_verified": "verified",
            "numerically_stable": "failed",
            "tokenizer_compatibility": {
                "status": "checked",
                "compatible": True,
            },
        }
    )

    assert isinstance(normalized, dict)
    assert normalized["eval_integrity_verified"] is True
    assert normalized["numerically_stable"] is False
    tokenizer = normalized["tokenizer_compatibility"]
    assert isinstance(tokenizer, dict)
    assert tokenizer["tokenizer_compatible"] is True


def test_approval_gated_dataset_is_selected_but_not_auto_approved() -> None:
    request = DatasetSearchRequest(
        request_id="dataset-request-1",
        diagnosis_report_id="diagnosis-1",
        problem_statement="Improve instruction following coverage",
        capability_targets=["instruction following"],
        required_languages=["en"],
        required_domains=["general"],
        required_formats=["jsonl"],
        provider_allowlist=["local_fixture"],
    )
    candidate = DatasetCandidate(
        candidate_id="dataset-candidate-1",
        provider_id="local_fixture",
        dataset_id="fixture/instructions",
        revision="v1",
        task_types=["instruction following"],
        languages=["en"],
        domains=["general"],
        format_profile={"record_format": "jsonl"},
        estimated_rows=100,
        estimated_bytes=10_000,
        license=None,
        provenance={"source": "fixture"},
        trust_level="local_fixture",
        compatibility={"compatible": True, "local_readable": True},
        artifact_ref="fixture.jsonl",
        evidence_refs=["evidence://fixture"],
    )

    decision = ApprovalAwareDatasetSelectionService().select(request, [candidate])

    assert decision.status == "selected"
    assert decision.selected_candidate_ids == [candidate.candidate_id]
    assert any(item.startswith("unknown_license:") for item in decision.required_approvals)
    assert decision.metadata["approval_gate"] == "required_before_acquisition"
    assert all(not issue.blocking for issue in decision.issues)


@dataclass
class _BrokenControlService:
    def launch(self, proposal: ExperimentProposal) -> TrainingRunHandle:
        raise AssertionError("not used")

    def status(self, run_id: str) -> TrainingRunHandle:
        return TrainingRunHandle(
            run_id=run_id,
            experiment_id="experiment-1",
            backend_id="fixture",
            status="interrupted",
        )

    def control(self, request: TrainingControlRequest) -> TrainingRunHandle:
        raise ValueError("malformed resume token")


def test_training_control_corruption_becomes_structured_failure() -> None:
    guarded = GuardedTrainingLifecycleService(_BrokenControlService())
    handle = guarded.control(
        TrainingControlRequest(
            request_id="control-1",
            run_id="run-1",
            action="resume",
            requested_by="operator",
            reason="continue bounded run",
        )
    )

    assert handle.status == "failed"
    assert any(issue.code == "training_control_evidence_invalid" for issue in handle.issues)


@dataclass
class _RecordingEvaluator:
    proposal: ExperimentProposal | None = None
    runs: list[TrainingRunHandle] = field(default_factory=list)

    def compare(
        self, proposal: ExperimentProposal, runs: list[TrainingRunHandle]
    ) -> ExperimentComparison:
        self.proposal = proposal
        self.runs = list(runs)
        return ExperimentComparison(
            comparison_id="comparison-1",
            experiment_id=proposal.experiment_id,
            baseline_run_id=proposal.baseline_ref,
            candidate_run_ids=[run.run_id for run in runs if run.run_id != proposal.baseline_ref],
            primary_outcome="equivalent_within_evidence",
        )


@dataclass
class _RecordingTrainingService:
    launches: int = 0

    def launch(self, proposal: ExperimentProposal) -> TrainingRunHandle:
        self.launches += 1
        return TrainingRunHandle(
            run_id=proposal.run_id,
            experiment_id=proposal.experiment_id,
            backend_id="fake",
            status="completed",
        )

    def status(self, run_id: str) -> TrainingRunHandle:
        raise AssertionError("not used")

    def control(self, request: TrainingControlRequest) -> TrainingRunHandle:
        raise AssertionError("not used")


def _coordinator(evaluator: object, training: object) -> AutonomousExperimentCoordinator:
    return AutonomousExperimentCoordinator(
        diagnosis_service=None,  # type: ignore[arg-type]
        planner=ClosedLoopExperimentPlanner(),
        dataset_registry=DatasetProviderRegistry(),
        dataset_selector=ApprovalAwareDatasetSelectionService(),
        model_providers={},
        model_selector=DeterministicModelSelectionService(),
        training_service=GuardedTrainingLifecycleService(training),  # type: ignore[arg-type]
        evaluation_service=evaluator,  # type: ignore[arg-type]
        record_sink=InMemoryIntegrationRecordSink(),
    )


def test_generic_baseline_reference_is_resolved_to_run_identity() -> None:
    evaluator = _RecordingEvaluator()
    coordinator = _coordinator(evaluator, _RecordingTrainingService())
    proposal = ExperimentProposal(
        experiment_id="experiment-1",
        run_id="candidate-run",
        lineage_id="lineage-1",
        stage_name="smoke_test",
        diagnosis_report_id="diagnosis-1",
        intervention_id="intervention-1",
        primary_variable="learning_rate",
        baseline_ref="checkpoint://stable",
    )
    candidate = TrainingRunHandle(
        run_id="candidate-run",
        experiment_id="experiment-1",
        backend_id="fake",
        status="completed",
    )
    baseline = TrainingRunHandle(
        run_id="baseline-run",
        experiment_id="baseline-experiment",
        backend_id="fake",
        status="completed",
    )

    comparison = coordinator.compare(
        proposal,
        [candidate],
        baseline_resolver=lambda ref: baseline if ref == "checkpoint://stable" else None,
    )

    assert comparison.baseline_run_id == "baseline-run"
    assert evaluator.proposal is not None
    assert evaluator.proposal.baseline_ref == "baseline-run"
    assert {run.run_id for run in evaluator.runs} == {"candidate-run", "baseline-run"}


def test_launch_requires_all_named_approval_evidence() -> None:
    training = _RecordingTrainingService()
    coordinator = _coordinator(_RecordingEvaluator(), training)
    proposal = ExperimentProposal(
        experiment_id="experiment-1",
        run_id="run-1",
        lineage_id="lineage-1",
        stage_name="smoke_test",
        diagnosis_report_id="diagnosis-1",
        intervention_id="intervention-1",
        primary_variable="dataset_mixture",
        required_approvals=["license_review"],
        status="ready",
    )

    blocked = coordinator.launch_approved(proposal, {})
    assert blocked.status == "failed"
    assert blocked.metadata["launch_attempted"] is False
    assert training.launches == 0

    launched = coordinator.launch_approved(
        proposal, {"license_review": "approval://license/1"}
    )
    assert launched.status == "completed"
    assert training.launches == 1


def test_unstructured_model_problem_does_not_reject_every_candidate() -> None:
    coordinator = _coordinator(_RecordingEvaluator(), _RecordingTrainingService())
    diagnosis = DiagnosisReport(
        report_id="diagnosis-1",
        request_id="request-1",
        run_id="run-1",
        lineage_id="lineage-1",
        stage_name="smoke_test",
        status="completed",
    )
    request = ModelSearchRequest(
        request_id="model-request-1",
        diagnosis_report_id=diagnosis.report_id,
        problem_statement="The model family may be limiting the target behavior",
        task_requirements=["The model family may be limiting the target behavior"],
        provider_allowlist=["catalog"],
    )

    normalized, issues = coordinator._normalize_model_request(diagnosis, request)

    assert normalized.task_requirements == []
    assert any(issue.code == "model_task_requirements_unstructured" for issue in issues)
