"""Actual staged orchestrator <-> production loop integration, CPU only.

The operation router is explicitly a test double. These tests verify the real
production loop, real staged orchestration, durable SQLite state and Judge
action ownership. They do not claim to test model-backed inference.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from hephaestus.control.staged_state import StagedAutonomousDependencies, StagedOperationRequest, StagedOperationResult
from hephaestus.production.composition import ProductionCompositionRoot, ProductionCompositionSettings
from hephaestus.production.loop import ProductionLoopRunner
from hephaestus.production.staged_driver import StagedProductionCycleDriver
from hephaestus.production.state import ACTION_EXECUTIONS

from test_staged_autonomous_orchestration import FakeOperationRouter, _services


@dataclass
class CompleteEvidenceRouter(FakeOperationRouter):
    """Supply the output identities required by the real production driver."""

    def execute(self, request: StagedOperationRequest) -> StagedOperationResult:
        result = super().execute(request)
        metadata = dict(result.metadata)
        if request.substep == "experiment_proposal":
            metadata["experiment_id"] = "evaluator-recovery-experiment-1"
        elif request.substep == "semantic_comparison":
            metadata["comparison_id"] = "comparison-verified-1"
        elif request.substep == "governed_verdict":
            metadata["confidence"] = 0.8
        elif request.substep == "promotion_gate":
            metadata["certification_state"] = "certification_not_eligible"
        return replace(result, metadata=metadata)


def _runtime(tmp_path):
    return ProductionCompositionRoot(
        ProductionCompositionSettings(
            state_root=tmp_path / "state",
            artifact_root=tmp_path / "artifacts",
        )
    ).build()


def _deps(router):
    def factory(runtime, state, cycle):
        services = _services(router)
        services.action_executor = None  # only production loop mutates lineage
        return StagedAutonomousDependencies(
            services=services,
            state_repository=runtime.state_repository,
        )
    return factory


def test_completed_staged_cycle_is_applied_once_by_production_loop(tmp_path):
    runtime = _runtime(tmp_path)
    router = CompleteEvidenceRouter()
    driver = StagedProductionCycleDriver(_deps(router))
    result = ProductionLoopRunner(runtime, driver, maximum_cycles=1).run(
        program_id="evaluator-repair",
        lineage_id="evaluator-lineage",
        stage_name="smoke_test",
    )
    assert result.status == "stopped" and result.cycle_index == 1
    assert result.latest_comparison_ref == "comparison-verified-1"
    assert router.action_calls == 0  # staged action intent never calls the mutating router
    recorded = runtime.state_repository.all(ACTION_EXECUTIONS)
    assert len(recorded) == 1
    assert recorded[0]["effective_action"] == "continue_lineage_best"
    again = ProductionLoopRunner(runtime, driver, maximum_cycles=1).run(
        program_id="evaluator-repair", lineage_id="evaluator-lineage", stage_name="smoke_test"
    )
    assert again.cycle_index == 1
    assert len(runtime.state_repository.all(ACTION_EXECUTIONS)) == 1


def test_training_wait_persists_and_restart_does_not_repeat_paid_launch(tmp_path):
    runtime = _runtime(tmp_path)
    first = CompleteEvidenceRouter(poll_statuses=["running"])
    one = ProductionLoopRunner(runtime, StagedProductionCycleDriver(_deps(first)), maximum_cycles=1).run(
        program_id="resume-repair", lineage_id="evaluator-lineage", stage_name="smoke_test"
    )
    assert one.status == "waiting" and one.cycle_index == 0
    assert one.metadata["pending_cycle"]["status"] == "pending"
    assert first.launch_calls == 1
    assert runtime.state_repository.all(ACTION_EXECUTIONS) == []

    # Recreate the controller and service objects, as after a process restart.
    second = CompleteEvidenceRouter(poll_statuses=["completed"])
    restarted = _runtime(tmp_path)
    two = ProductionLoopRunner(
        restarted, StagedProductionCycleDriver(_deps(second)), maximum_cycles=1
    ).run(program_id="resume-repair", lineage_id="evaluator-lineage", stage_name="smoke_test")
    assert two.cycle_index == 1 and two.status == "stopped"
    assert second.launch_calls == 0  # the persisted lifecycle_launch is skipped
    assert second.comparison_calls == 1
    assert len(restarted.state_repository.all(ACTION_EXECUTIONS)) == 1


def test_nonresumable_staged_failure_never_executes_judge_action(tmp_path):
    runtime = _runtime(tmp_path)
    router = CompleteEvidenceRouter(fail_substep="model_selection")
    result = ProductionLoopRunner(
        runtime, StagedProductionCycleDriver(_deps(router)), maximum_cycles=1
    ).run(program_id="failed-repair", lineage_id="evaluator-lineage", stage_name="smoke_test")
    assert result.status == "blocked" and result.cycle_index == 0
    assert runtime.state_repository.all(ACTION_EXECUTIONS) == []


def test_driver_rejects_second_side_effecting_staged_executor(tmp_path):
    runtime = _runtime(tmp_path)
    router = CompleteEvidenceRouter()
    def unsafe(runtime, state, cycle):
        return StagedAutonomousDependencies(
            services=_services(router),
            state_repository=runtime.state_repository,
        )
    with pytest.raises(ValueError, match="duplicate"):
        ProductionLoopRunner(
            runtime, StagedProductionCycleDriver(unsafe), maximum_cycles=1
        ).run(program_id="unsafe", lineage_id="evaluator-lineage", stage_name="smoke_test")
    assert runtime.state_repository.all(ACTION_EXECUTIONS) == []
