from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from hephaestus.control.spine import SPINE_ORDER
from hephaestus.production.actions import GovernedActionExecutor
from hephaestus.production.composition import ProductionCompositionRoot, ProductionCompositionSettings
from hephaestus.production.loop import ProductionCycleResult, ProductionLoopRunner
from hephaestus.production.recovery import InfrastructureRecoveryController, RecoverableInfrastructureError
from hephaestus.schemas.lineage_state import LineageState
from hephaestus.state.lineage_store import LineageStore
from hephaestus.storage.sqlite import SQLiteStateRepository


def _phases(tag: str) -> dict[str, list[str]]:
    return {phase.value: [f"evidence://{tag}/{phase.value}"] for phase in SPINE_ORDER}


def test_production_composition_root_supplies_real_services(tmp_path: Path) -> None:
    runtime = ProductionCompositionRoot(
        ProductionCompositionSettings(
            state_root=tmp_path / "state",
            artifact_root=tmp_path / "artifacts",
        )
    ).build()
    inventory = runtime.service_inventory()
    assert runtime.state_repository.ready()
    assert set(inventory) == {
        "state_repository",
        "artifact_store",
        "secrets_provider",
        "diagnosis",
        "planner",
        "dataset_discovery",
        "dataset_selection",
        "dataset_acquisition",
        "preprocessing",
        "model_selection",
        "training_lifecycle",
        "evaluator",
        "judge",
        "scientific_recovery",
        "infrastructure_recovery",
        "action_executor",
    }
    assert all(value and "Fake" not in value and "InMemory" not in value for value in inventory.values())


def test_infrastructure_recovery_retries_only_before_scientific_progress(tmp_path: Path) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.sqlite3")
    recovery = InfrastructureRecoveryController(repository, maximum_attempts=4)
    calls = 0

    def eventually_works(attempt: int) -> str:
        nonlocal calls
        calls += 1
        if attempt < 3:
            raise RecoverableInfrastructureError("capacity_unavailable", "no GPU capacity")
        return "ok"

    assert recovery.run("capacity-op", eventually_works) == "ok"
    assert calls == 3
    rows = [row for row in repository.all("production_infrastructure_recovery") if row["operation_id"] == "capacity-op"]
    assert [row["status"] for row in rows] == ["retrying", "retrying", "completed"]
    assert all(row["scientific_variables_changed"] is False for row in rows)

    with pytest.raises(RecoverableInfrastructureError):
        recovery.run(
            "unsafe-op",
            lambda _attempt: (_ for _ in ()).throw(
                RecoverableInfrastructureError(
                    "worker_lost_before_progress",
                    "worker lost after progress",
                    optimizer_steps=1,
                )
            ),
        )
    unsafe = repository.get_latest("production_infrastructure_recovery", "operation_id", "unsafe-op")
    assert unsafe is not None and unsafe["status"] == "blocked"


def test_governed_action_executor_applies_reject_branch_rollback_restart_and_promotion(tmp_path: Path) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.sqlite3")
    lineage = LineageStore(tmp_path)
    lineage.set_current(
        LineageState(
            lineage_id="lineage-main",
            stage_name="smoke_test",
            status="active",
            best_checkpoint_ref="checkpoint://old",
            last_stable_checkpoint_ref="checkpoint://stable",
        ).to_dict()
    )
    executor = GovernedActionExecutor(tmp_path, repository)

    reject = executor.apply(
        lineage_id="lineage-main",
        run_id="bad-run",
        requested_action="reject_checkpoint",
    )
    assert reject["effective_action"] == "reject_candidate"
    assert "bad-run" in LineageStore(tmp_path).get_current("lineage-main")["recent_failures"]

    branch = executor.apply(
        lineage_id="lineage-main",
        run_id="branch-run",
        requested_action="branch_new_experiment",
        checkpoint_ref="checkpoint://old",
        approval_status="approved",
        approval_ref="approval://branch",
        promotion_allowed=True,
    )
    child = branch["after"]["metadata"]["latest_child_lineage_id"]
    assert LineageStore(tmp_path).get_parent(child) == "lineage-main"

    rollback = executor.apply(
        lineage_id="lineage-main",
        run_id="rollback-run",
        requested_action="rollback_to_checkpoint",
        checkpoint_ref="checkpoint://stable",
        approval_status="approved",
        approval_ref="approval://rollback",
        promotion_allowed=True,
    )
    assert rollback["after"]["best_checkpoint_ref"] == "checkpoint://stable"

    restart = executor.apply(
        lineage_id="lineage-main",
        run_id="restart-run",
        requested_action="restart_lineage",
        checkpoint_ref="checkpoint://stable",
        approval_status="approved",
        approval_ref="approval://restart",
        promotion_allowed=True,
    )
    assert restart["after"]["status"] == "restarted"

    promoted = executor.apply(
        lineage_id="lineage-main",
        run_id="good-run",
        requested_action="promote_checkpoint",
        checkpoint_ref="checkpoint://good",
        approval_status="approved",
        approval_ref="approval://promotion",
        promotion_allowed=True,
        certification_state="certification_passed",
        confidence=0.95,
    )
    assert promoted["after"]["best_checkpoint_ref"] == "checkpoint://good"
    assert promoted["after"]["certified_stable_checkpoint_ref"] == "checkpoint://good"
    assert promoted["after"]["status"] == "stable"
    assert executor.apply(
        lineage_id="lineage-main",
        run_id="good-run",
        requested_action="promote_checkpoint",
        checkpoint_ref="checkpoint://good",
        approval_status="approved",
        approval_ref="approval://promotion",
        promotion_allowed=True,
        certification_state="certification_passed",
        confidence=0.95,
    )["execution_id"] == promoted["execution_id"]


@dataclass
class _TwoCycleDriver:
    transient_failures: int = 2
    calls: int = 0

    def execute_cycle(self, *, runtime, state, cycle_index: int) -> ProductionCycleResult:  # noqa: ANN001
        self.calls += 1
        if self.calls <= self.transient_failures:
            raise RecoverableInfrastructureError("capacity_unavailable", "temporary GPU capacity")
        if cycle_index == 1:
            return ProductionCycleResult(
                cycle_id="cycle-1",
                run_id="candidate-1",
                experiment_id="experiment-1",
                status="completed",
                judge_action="reject_checkpoint",
                confidence=0.7,
                comparison_ref="comparison://1",
                phase_evidence=_phases("cycle-1"),
                evidence={"semantic_outcome": "regressed", "diagnosis_input": "comparison://1"},
                next_cycle={"intervention": "change_model", "reason": "prior candidate rejected"},
            )
        return ProductionCycleResult(
            cycle_id="cycle-2",
            run_id="candidate-2",
            experiment_id="experiment-2",
            status="completed",
            judge_action="promote_checkpoint",
            checkpoint_ref="checkpoint://candidate-2",
            confidence=0.95,
            promotion_allowed=True,
            certification_state="certification_passed",
            approval_status="approved",
            approval_ref="approval://human-review-and-promotion",
            comparison_ref="comparison://2",
            phase_evidence=_phases("cycle-2"),
            evidence={"semantic_outcome": "improved"},
        )


def test_loop_automatically_recovers_continues_and_applies_final_promotion(tmp_path: Path) -> None:
    runtime = ProductionCompositionRoot(
        ProductionCompositionSettings(
            state_root=tmp_path / "state",
            artifact_root=tmp_path / "artifacts",
            maximum_infrastructure_attempts=5,
        )
    ).build()
    driver = _TwoCycleDriver()
    final = ProductionLoopRunner(runtime, driver, maximum_cycles=4).run(
        program_id="program-1",
        lineage_id="lineage-main",
        stage_name="smoke_test",
    )
    assert final.status == "completed"
    assert final.stop_reason == "candidate_promoted"
    assert final.cycle_index == 2
    assert final.completed_cycles == ["cycle-1", "cycle-2"]
    assert driver.calls == 4  # two infrastructure retries + two successful cycles
    stored = runtime.loop_state.get("program-1")
    assert stored is not None and stored.status == "completed"
    lineage = LineageStore(tmp_path / "state").get_current("lineage-main")
    assert lineage is not None
    assert lineage["certified_stable_checkpoint_ref"] == "checkpoint://candidate-2"
    assert runtime.state_repository.all("production_infrastructure_recovery")


def test_completed_cycle_requires_all_eight_control_spine_phases(tmp_path: Path) -> None:
    runtime = ProductionCompositionRoot(
        ProductionCompositionSettings(state_root=tmp_path / "state", artifact_root=tmp_path / "artifacts")
    ).build()

    class MissingPhase:
        def execute_cycle(self, *, runtime, state, cycle_index):  # noqa: ANN001
            return ProductionCycleResult(
                cycle_id="bad",
                run_id="bad",
                experiment_id="bad",
                status="completed",
                judge_action="reject_checkpoint",
                phase_evidence={SPINE_ORDER[0].value: ["only-one-phase"]},
            )

    with pytest.raises(ValueError, match="mandatory phase evidence"):
        ProductionLoopRunner(runtime, MissingPhase(), maximum_cycles=1).run(
            program_id="bad-program", lineage_id="lineage", stage_name="smoke_test"
        )
