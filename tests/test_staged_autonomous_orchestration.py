from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from hephaestus.control.orchestrator import build_orchestrator
from hephaestus.control.spine import SPINE_ORDER, SpinePhase
from hephaestus.control.staged_autonomous import (
    PHASE_SUBSTEPS,
    WORKFLOW_RECORD_COLLECTION,
    StagedApprovalDecision,
    StagedApprovalRequest,
    StagedAutonomousDependencies,
    StagedAutonomousServices,
    StagedOperationRequest,
    StagedOperationResult,
    StagedOutputRecord,
)
from hephaestus.storage.filesystem import JsonLineStateRepository


_RECORD_KIND = {
    "judge_entry_decision": "judge_entry",
    "evidence_collection": "evidence_bundle",
    "evidence_based_diagnosis": "diagnosis_report",
    "readiness_to_plan": "planning_readiness",
    "diagnosis_handoff": "diagnosis_handoff",
    "intervention_ranking": "intervention_proposal",
    "experiment_proposal": "experiment_proposal",
    "discovery_request_preparation": "search_request",
    "approval_requirement_discovery": "approval_requirements",
    "dataset_discovery": "dataset_discovery_result",
    "dataset_selection": "dataset_selection_decision",
    "dataset_acquisition": "acquisition_receipt",
    "license_provenance_evidence": "license_provenance_evidence",
    "dataset_audit": "dataset_audit",
    "manifest_production": "dataset_manifest",
    "approved_source_preprocessing": "preprocessing_report",
    "deduplication_contamination": "deduplication_contamination_evidence",
    "tokenizer_compatibility": "tokenizer_compatibility",
    "trainable_data_contract": "trainable_data_contract",
    "model_discovery": "model_discovery_result",
    "model_selection": "model_selection_decision",
    "training_input_binding": "training_input_binding",
    "launch_configuration": "launch_configuration",
    "lifecycle_launch": "training_run_handle",
    "training_status_poll": "training_run_handle",
    "runtime_observation": "runtime_observation",
    "runtime_evidence": "runtime_evidence",
    "bounded_recovery_advice": "recovery_decision",
    "runtime_control_governance": "runtime_control_recommendation",
    "checkpoint_resolution": "checkpoint_resolution",
    "generation_prompt_materialization": "generation_prompt_manifest",
    "baseline_generation": "generation_report",
    "candidate_generation": "generation_report",
    "semantic_comparison": "experiment_comparison",
    "deterministic_regression_evidence": "deterministic_regression_evidence",
    "repeatability_variance_evidence": "repeatability_variance_evidence",
    "human_review_references": "human_review_references",
    "governed_verdict": "judge_exit",
    "promotion_gate": "promotion_gate_report",
    "action_application": "action_decision",
}


@dataclass
class FakeOperationRouter:
    action: str = "continue_lineage_best"
    dataset_approvals: list[str] = field(default_factory=list)
    model_approvals: list[str] = field(default_factory=list)
    poll_statuses: list[str] = field(default_factory=lambda: ["completed"])
    missing_diagnosis_evidence: list[str] = field(default_factory=list)
    generation_settings: tuple[str, str] = ("decode-v1", "decode-v1")
    checkpoint_ref: str = "checkpoint://candidate/1"
    fail_substep: str | None = None
    omit_record_substep: str | None = None
    forced_statuses: dict[str, str] = field(default_factory=dict)
    deterministic_passed: bool = True
    promotion_allowed: bool = True
    calls: list[tuple[str, str, int]] = field(default_factory=list)
    launch_calls: int = 0
    acquisition_calls: int = 0
    action_calls: int = 0
    comparison_calls: int = 0
    applied_actions: list[str] = field(default_factory=list)

    def execute(self, request: StagedOperationRequest) -> StagedOperationResult:
        self.calls.append((request.phase, request.substep, request.attempt))
        if request.substep == self.fail_substep:
            raise RuntimeError("fixture subsystem failure")

        metadata: dict[str, object] = {}
        output_refs = (f"evidence://{request.substep}/{request.attempt}",)
        if request.substep == "evidence_based_diagnosis":
            metadata["missing_evidence"] = list(self.missing_diagnosis_evidence)
        elif request.substep == "dataset_selection":
            metadata.update(
                required_approvals=list(self.dataset_approvals),
                subject_ref="dataset-selection://1",
                dataset_revision="dataset-revision-v1",
                planner_authorized=True,
            )
        elif request.substep == "dataset_acquisition":
            self.acquisition_calls += 1
        elif request.substep == "model_selection":
            metadata.update(
                required_approvals=list(self.model_approvals),
                subject_ref="model-selection://1",
                model_revision="model-revision-v1",
            )
        elif request.substep == "launch_configuration":
            metadata["config_identity"] = "config-sha256-v1"
        elif request.substep == "lifecycle_launch":
            self.launch_calls += 1
            metadata.update(training_status="running", seed_identity="seeds-11-29-47")
        elif request.substep == "training_status_poll":
            index = min(
                sum(1 for _, substep, _ in self.calls if substep == "training_status_poll") - 1,
                len(self.poll_statuses) - 1,
            )
            metadata["training_status"] = self.poll_statuses[index]
        elif request.substep == "checkpoint_resolution":
            metadata.update(checkpoint_ref=self.checkpoint_ref, eval_pack_id="semantic_behavior:1.0.0")
            output_refs = (self.checkpoint_ref,)
        elif request.substep == "baseline_generation":
            metadata["generation_settings_id"] = self.generation_settings[0]
        elif request.substep == "candidate_generation":
            metadata["generation_settings_id"] = self.generation_settings[1]
        elif request.substep == "semantic_comparison":
            self.comparison_calls += 1
            metadata.update(
                comparison_outcome="improved",
                proposed_action="promote_checkpoint",  # advisory and deliberately ignored
            )
        elif request.substep == "deterministic_regression_evidence":
            metadata["deterministic_passed"] = self.deterministic_passed
        elif request.substep == "governed_verdict":
            metadata.update(proposed_action=self.action, subject_ref=self.checkpoint_ref)
        elif request.substep == "promotion_gate":
            metadata.update(
                promotion_allowed=self.promotion_allowed,
                action_allowed=self.promotion_allowed,
            )
        elif request.substep == "action_application":
            self.action_calls += 1
            boundary = request.prior_outputs.get("judge_exit.action_boundary", {})
            applied = str(boundary.get("proposed_action", ""))
            self.applied_actions.append(applied)
            metadata["applied_action"] = applied

        kind = _RECORD_KIND.get(request.substep, request.substep)
        records = () if request.substep == self.omit_record_substep else (
            StagedOutputRecord(
                kind,
                {"run_id": request.run_id, "operation_id": request.operation_id, **metadata},
            ),
        )
        return StagedOperationResult(
            status=self.forced_statuses.get(request.substep, "completed"),
            output_refs=output_refs,
            records=records,
            metadata=metadata,
        )


@dataclass
class FakeApprovalService:
    decisions: dict[str, StagedApprovalDecision] = field(default_factory=dict)
    requests: list[StagedApprovalRequest] = field(default_factory=list)
    mismatch: bool = False

    def decision_for(self, request: StagedApprovalRequest) -> StagedApprovalDecision | None:
        self.requests.append(request)
        if self.mismatch:
            return StagedApprovalDecision(
                request_id=request.request_id,
                operation_id=request.operation_id,
                subject_ref="stale://subject",
                status="approved",
                approval_ref="approval://stale",
                requirements=request.requirements,
            )
        return self.decisions.get(request.request_id)

    def approve_latest(self) -> None:
        request = self.requests[-1]
        self.decisions[request.request_id] = StagedApprovalDecision(
            request_id=request.request_id,
            operation_id=request.operation_id,
            subject_ref=request.subject_ref,
            status="approved",
            approval_ref=f"approval://{request.request_id}",
            requirements=request.requirements,
        )


@dataclass
class FakeArtifactStore:
    verified: bool

    def verify(self, artifact_ref: str) -> bool:
        return self.verified


def _services(router: FakeOperationRouter) -> StagedAutonomousServices:
    return StagedAutonomousServices(
        judge_entry=router,
        evidence_collector=router,
        diagnosis=router,
        plan_readiness=router,
        planner=router,
        dataset_discovery=router,
        dataset_selection=router,
        dataset_acquisition=router,
        data_preprocessor=router,
        model_discovery=router,
        model_selection=router,
        training_lifecycle=router,
        runtime_monitor=router,
        generation=router,
        evaluator=router,
        recovery=router,
        judge_exit=router,
        action_executor=router,
    )


def _orchestrator(
    tmp_path: Path,
    router: FakeOperationRouter,
    approvals: FakeApprovalService | None = None,
    artifact_store: object | None = None,
):
    repository = JsonLineStateRepository(tmp_path / "staged-state")
    dependencies = StagedAutonomousDependencies(
        services=_services(router),
        state_repository=repository,
        approval_service=approvals,
        artifact_store=artifact_store,  # type: ignore[arg-type]
    )
    orchestrator = build_orchestrator(
        state_root=tmp_path / "legacy-unused",
        run_id="run-staged",
        lineage_id="lineage-staged",
        stage_name="smoke_test",
        mode="governed_autonomous",
        staged_dependencies=dependencies,
    )
    return orchestrator, repository


def _records(repository: JsonLineStateRepository) -> list[dict[str, object]]:
    return repository.all(WORKFLOW_RECORD_COLLECTION)


def test_successful_staged_run_preserves_exact_spine_and_substep_order(tmp_path: Path) -> None:
    router = FakeOperationRouter()
    orchestrator, repository = _orchestrator(tmp_path, router)

    state = orchestrator.run("run-staged")

    assert state.status == "completed"
    assert state.completion_marker is True
    assert state.phase_order == [phase.value for phase in SPINE_ORDER]
    observed = [
        (str(row["phase"]), str(row["substep"]))
        for row in _records(repository)
        if row["kind"] == "substep_result"
    ]
    expected = [
        (phase.value, substep)
        for phase in SPINE_ORDER
        for substep in PHASE_SUBSTEPS[phase]
    ]
    assert observed == expected

    positions = {substep: index for index, (_, substep) in enumerate(observed)}
    assert positions["evidence_based_diagnosis"] < positions["intervention_ranking"]
    assert positions["dataset_discovery"] < positions["manifest_production"]
    assert positions["model_selection"] < positions["lifecycle_launch"]
    assert positions["candidate_generation"] < positions["semantic_comparison"]


def test_decision_critical_records_and_replay_evidence_are_persisted_in_order(tmp_path: Path) -> None:
    orchestrator, repository = _orchestrator(tmp_path, FakeOperationRouter())
    state = orchestrator.run()
    rows = _records(repository)

    assert [int(row["sequence"]) for row in rows] == list(range(1, len(rows) + 1))
    kinds = {str(row["kind"]) for row in rows}
    assert {
        "judge_entry",
        "diagnosis_report",
        "intervention_proposal",
        "search_request",
        "dataset_selection_decision",
        "acquisition_receipt",
        "dataset_manifest",
        "preprocessing_report",
        "trainable_data_contract",
        "model_selection_decision",
        "training_run_handle",
        "generation_report",
        "experiment_comparison",
        "judge_exit",
        "action_decision",
        "recovery_decision",
        "replay_metadata",
    }.issubset(kinds)
    replay = state.steps["judge_exit.replay_evidence"].output
    assert replay["replay_complete"] is True
    assert replay["phase_order"] == [phase.value for phase in SPINE_ORDER]
    assert replay["dataset_revision"] == "dataset-revision-v1"
    assert replay["model_revision"] == "model-revision-v1"


def test_missing_diagnosis_evidence_blocks_before_planner(tmp_path: Path) -> None:
    router = FakeOperationRouter(missing_diagnosis_evidence=["verified_eval_integrity"])
    orchestrator, _ = _orchestrator(tmp_path, router)

    state = orchestrator.run()

    assert state.status == "inconclusive"
    assert state.current_substep == "readiness_to_plan"
    assert "diagnosis_evidence_insufficient" in state.blocking_issues
    assert all(phase != SpinePhase.PLANNER.value for phase, _, _ in router.calls)


def test_dataset_and_model_approval_pause_resume_without_duplicate_work(tmp_path: Path) -> None:
    router = FakeOperationRouter(
        dataset_approvals=["dataset_license_approval"],
        model_approvals=["model_risk_approval"],
    )
    approvals = FakeApprovalService()
    orchestrator, repository = _orchestrator(tmp_path, router, approvals)

    first = orchestrator.run()
    assert first.status == "approval_pending"
    assert first.current_substep == "acquisition_approval_gate"
    approvals.approve_latest()

    second = orchestrator.resume()
    assert second.status == "approval_pending"
    assert second.current_substep == "model_approval_gate"
    acquisition_calls = router.acquisition_calls
    approvals.approve_latest()

    completed = orchestrator.resume()
    assert completed.status == "completed"
    assert router.acquisition_calls == acquisition_calls
    assert router.launch_calls == 1
    requests = [row for row in _records(repository) if row["kind"] == "approval_request"]
    assert len(requests) == 2


def test_stale_or_mismatched_approval_is_rejected(tmp_path: Path) -> None:
    router = FakeOperationRouter(dataset_approvals=["dataset_license_approval"])
    orchestrator, _ = _orchestrator(tmp_path, router, FakeApprovalService(mismatch=True))

    state = orchestrator.run()

    assert state.status == "blocked"
    assert state.resumable is False
    assert any(issue.startswith("stale_or_mismatched_approval") for issue in state.blocking_issues)
    assert router.acquisition_calls == 0


@pytest.mark.parametrize("first_status", ["running", "interrupted"])
def test_training_is_polled_asynchronously_without_duplicate_launch(
    tmp_path: Path, first_status: str
) -> None:
    router = FakeOperationRouter(poll_statuses=[first_status, "completed"])
    orchestrator, _ = _orchestrator(tmp_path, router)

    waiting = orchestrator.run()
    assert waiting.status == "interrupted"
    assert waiting.current_substep == "training_status_poll"
    assert router.launch_calls == 1

    completed = orchestrator.resume()
    assert completed.status == "completed"
    assert router.launch_calls == 1
    assert sum(1 for _, step, _ in router.calls if step == "training_status_poll") == 2


def test_cancelled_training_propagates_cancelled_state(tmp_path: Path) -> None:
    router = FakeOperationRouter(poll_statuses=["cancelled"])
    orchestrator, _ = _orchestrator(tmp_path, router)

    state = orchestrator.run()

    assert state.status == "cancelled"
    assert state.resumable is False
    assert state.current_substep == "training_status_poll"
    assert all(step != "checkpoint_resolution" for _, step, _ in router.calls)


def test_retryable_failure_keeps_stable_operation_identity(tmp_path: Path) -> None:
    router = FakeOperationRouter(forced_statuses={"dataset_discovery": "retryable_failure"})
    orchestrator, _ = _orchestrator(tmp_path, router)

    first = orchestrator.run()
    first_step = first.steps["data_acquisition_audit.dataset_discovery"]
    router.forced_statuses.clear()
    completed = orchestrator.resume()
    second_step = completed.steps["data_acquisition_audit.dataset_discovery"]

    assert completed.status == "completed"
    assert second_step.operation_id == first_step.operation_id
    assert second_step.attempt == 2


def test_corrupted_checkpoint_blocks_evaluation_before_generation(tmp_path: Path) -> None:
    router = FakeOperationRouter(checkpoint_ref=f"sha256:{'a' * 64}")
    orchestrator, _ = _orchestrator(
        tmp_path, router, artifact_store=FakeArtifactStore(verified=False)
    )

    state = orchestrator.run()

    assert state.status == "blocked"
    assert "checkpoint_integrity_failed" in state.blocking_issues
    assert all(step != "baseline_generation" for _, step, _ in router.calls)


def test_generation_parity_mismatch_blocks_comparison(tmp_path: Path) -> None:
    router = FakeOperationRouter(generation_settings=("decode-a", "decode-b"))
    orchestrator, _ = _orchestrator(tmp_path, router)

    state = orchestrator.run()

    assert state.status == "blocked"
    assert state.current_substep == "semantic_comparison"
    assert "generation_parity_mismatch" in state.blocking_issues
    assert router.comparison_calls == 0


def test_deterministic_regression_reaches_judge_but_evaluator_cannot_promote(tmp_path: Path) -> None:
    router = FakeOperationRouter(deterministic_passed=False, action="reject_checkpoint")
    orchestrator, _ = _orchestrator(tmp_path, router)

    state = orchestrator.run()

    assert state.status == "completed"
    assert any(step == "governed_verdict" for _, step, _ in router.calls)
    assert router.applied_actions == ["reject_checkpoint"]
    assert "promote_checkpoint" not in router.applied_actions


def test_planner_authorization_does_not_bypass_dataset_approval(tmp_path: Path) -> None:
    router = FakeOperationRouter(dataset_approvals=["dataset_license_approval"])
    orchestrator, _ = _orchestrator(tmp_path, router)

    state = orchestrator.run()

    selection = state.steps["data_acquisition_audit.dataset_selection"].output
    assert selection["planner_authorized"] is True
    assert state.status == "approval_pending"
    assert router.acquisition_calls == 0


def test_high_impact_action_requires_approval_and_rollback_is_not_applied_twice(tmp_path: Path) -> None:
    router = FakeOperationRouter(action="rollback_to_checkpoint")
    approvals = FakeApprovalService()
    orchestrator, repository = _orchestrator(tmp_path, router, approvals)

    pending = orchestrator.run()
    assert pending.status == "approval_pending"
    assert pending.current_substep == "action_approval_gate"
    assert router.action_calls == 0
    approvals.approve_latest()

    completed = orchestrator.resume()
    assert completed.status == "completed"
    assert router.applied_actions == ["rollback_to_checkpoint"]
    orchestrator.resume()
    assert router.applied_actions == ["rollback_to_checkpoint"]
    assert len([row for row in _records(repository) if row["kind"] == "approval_request"]) == 1


def test_completed_workflow_resume_is_a_deterministic_noop(tmp_path: Path) -> None:
    router = FakeOperationRouter()
    orchestrator, repository = _orchestrator(tmp_path, router)
    completed = orchestrator.run()
    call_count = len(router.calls)
    record_count = len(_records(repository))

    repeated = orchestrator.resume()

    assert repeated.to_dict() == completed.to_dict()
    assert len(router.calls) == call_count
    assert len(_records(repository)) == record_count


def test_unknown_action_is_blocked_before_executor(tmp_path: Path) -> None:
    router = FakeOperationRouter(action="invented_unregistered_action")
    orchestrator, _ = _orchestrator(tmp_path, router)

    state = orchestrator.run()

    assert state.status == "blocked"
    assert state.current_substep == "action_boundary"
    assert "unknown_action" in state.blocking_issues
    assert router.action_calls == 0


def test_promotion_gate_blocks_before_action_executor(tmp_path: Path) -> None:
    router = FakeOperationRouter(action="promote_checkpoint", promotion_allowed=False)
    approvals = FakeApprovalService()
    orchestrator, _ = _orchestrator(tmp_path, router, approvals)

    pending = orchestrator.run()
    assert pending.current_substep == "action_approval_gate"
    approvals.approve_latest()
    blocked = orchestrator.resume()

    assert blocked.status == "blocked"
    assert "governed_action_gate_blocked" in blocked.blocking_issues
    assert router.action_calls == 0


def test_subsystem_exception_is_terminal_failure_not_success(tmp_path: Path) -> None:
    router = FakeOperationRouter(fail_substep="dataset_discovery")
    orchestrator, repository = _orchestrator(tmp_path, router)

    state = orchestrator.run()

    assert state.status == "terminal_failure"
    assert state.resumable is False
    assert "subsystem_exception:RuntimeError" in state.blocking_issues
    assert any(row["kind"] == "subsystem_failure" for row in _records(repository))


def test_missing_decision_critical_record_is_terminal_failure(tmp_path: Path) -> None:
    router = FakeOperationRouter(omit_record_substep="evidence_based_diagnosis")
    orchestrator, _ = _orchestrator(tmp_path, router)

    state = orchestrator.run()

    assert state.status == "terminal_failure"
    assert "decision_critical_record_missing:diagnosis_report" in state.blocking_issues


def test_legacy_orchestrator_remains_default_and_passes(tmp_path: Path) -> None:
    legacy = build_orchestrator(state_root=tmp_path / "legacy", run_id="legacy-run")

    results = legacy.run("legacy-run")

    assert [result.phase for result in results] == list(SPINE_ORDER)


def test_unknown_orchestration_mode_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown orchestration mode"):
        build_orchestrator(
            state_root=tmp_path,
            run_id="run-unknown-mode",
            mode="not-a-real-mode",
        )
