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
from hephaestus.state.lineage_store import LineageStore
from hephaestus.state._json_store import JsonSingleDocument
from hephaestus.schemas.lineage_state import LineageState

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
    recorded = [row for row in runtime.state_repository.all(ACTION_EXECUTIONS) if row["status"] == "applied"]
    assert len(recorded) == 1
    assert recorded[0]["effective_action"] == "continue_lineage_best"
    again = ProductionLoopRunner(runtime, driver, maximum_cycles=1).run(
        program_id="evaluator-repair", lineage_id="evaluator-lineage", stage_name="smoke_test"
    )
    assert again.cycle_index == 1
    assert len([r for r in runtime.state_repository.all(ACTION_EXECUTIONS) if r["status"] == "applied"]) == 1


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
    assert len([r for r in restarted.state_repository.all(ACTION_EXECUTIONS) if r["status"] == "applied"]) == 1


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


def test_prepared_judge_action_replays_exactly_once_after_controller_crash(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path)
    executor = runtime.action_executor
    original = LineageStore.set_current
    crashed = False

    def crash_after_lineage_write(self, record):
        nonlocal crashed
        original(self, record)
        if record["lineage_id"] == "evaluator-lineage" and not crashed:
            crashed = True
            raise RuntimeError("injected crash after lineage write, before action receipt")

    monkeypatch.setattr(LineageStore, "set_current", crash_after_lineage_write)
    args = {
        "lineage_id": "evaluator-lineage",
        "run_id": "first-evaluator-run",
        "requested_action": "reject_candidate",
    }
    with pytest.raises(RuntimeError, match="injected crash"):
        executor.apply(**args)
    prepared = runtime.state_repository.all(ACTION_EXECUTIONS)
    assert len(prepared) == 1 and prepared[0]["status"] == "prepared"
    assert LineageStore(runtime.settings.state_root).get_current("evaluator-lineage")["recent_failures"] == ["first-evaluator-run"]

    monkeypatch.setattr(LineageStore, "set_current", original)
    applied = executor.apply(**args)
    repeated = executor.apply(**args)
    assert applied["status"] == repeated["status"] == "applied"
    assert applied["execution_id"] == repeated["execution_id"]
    assert LineageStore(runtime.settings.state_root).get_current("evaluator-lineage")["recent_failures"] == ["first-evaluator-run"]
    assert [r["status"] for r in runtime.state_repository.all(ACTION_EXECUTIONS)] == ["prepared", "applied"]


def test_prepared_branch_reconciles_child_without_duplicate_parent_mutation(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path)
    root = runtime.settings.state_root
    lineage = LineageStore(root)
    lineage.set_current(LineageState(
        lineage_id="lineage-parent", stage_name="smoke_test",
        status="active", best_checkpoint_ref="checkpoint://parent"
    ).to_dict())
    executor = runtime.action_executor
    original = LineageStore.set_current

    def crash_before_parent_after_child(self, record):
        if record["lineage_id"] == "lineage-parent":
            raise RuntimeError("injected crash after child, before parent")
        return original(self, record)

    monkeypatch.setattr(LineageStore, "set_current", crash_before_parent_after_child)
    args = {
        "lineage_id": "lineage-parent", "run_id": "branch-test",
        "requested_action": "branch_new_experiment",
        "checkpoint_ref": "checkpoint://parent",
        "approval_status": "approved", "approval_ref": "approval://branch",
    }
    with pytest.raises(RuntimeError, match="injected crash"):
        executor.apply(**args)
    rows = runtime.state_repository.all(ACTION_EXECUTIONS)
    assert len(rows) == 1 and rows[0]["status"] == "prepared"
    child = rows[0]["child_after"]
    assert child is not None
    assert lineage.get_current(child["lineage_id"]) == child
    assert child["lineage_id"] not in lineage.get_current("lineage-parent")["child_lineage_ids"]

    monkeypatch.setattr(LineageStore, "set_current", original)
    result = executor.apply(**args)
    assert result["status"] == "applied"
    assert lineage.get_current("lineage-parent")["child_lineage_ids"] == [child["lineage_id"]]
    assert lineage.get_parent(child["lineage_id"]) == "lineage-parent"
    assert [r["status"] for r in runtime.state_repository.all(ACTION_EXECUTIONS)] == ["prepared", "applied"]


def test_atomic_lineage_file_write_preserves_previous_json_when_replace_fails(tmp_path, monkeypatch):
    document = JsonSingleDocument(tmp_path, "lineage_states.json")
    document.write({"lineage-1": {"status": "before"}})
    import hephaestus.state._json_store as store_module

    def crash_before_atomic_replace(src, dst):
        raise OSError("injected replace failure")

    monkeypatch.setattr(store_module.os, "replace", crash_before_atomic_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        document.write({"lineage-1": {"status": "after"}})
    assert document.read() == {"lineage-1": {"status": "before"}}
    assert not list(tmp_path.glob("*.partial"))
