from pathlib import Path

import pytest

from hephaestus.backends.dry_run_backend import DryRunBackend
from hephaestus.control.orchestrator import build_orchestrator
from hephaestus.control.spine import SPINE_ORDER
from hephaestus.evaluation.stage_interpreter import interpret_stage
from hephaestus.policy.judge_policy import JudgePolicy
from hephaestus.schemas.judge_exit import JudgeExitAction
from hephaestus.schemas.lineage_state import LineageState
from hephaestus.schemas.runtime_event import RuntimeEvent, RuntimeEventCategory
from hephaestus.state.decision_store import DecisionStore
from hephaestus.state.lineage_store import LineageStore
from hephaestus.state.run_store import RunStore


def test_schema_serialization_roundtrip() -> None:
    state = LineageState(
        lineage_id="l1",
        parent_lineage_id=None,
        status="active",
        stage_name="early_pretraining",
        latest_run_id="r1",
    )
    assert LineageState.from_dict(state.to_dict()).lineage_id == "l1"


def test_run_store_append_only(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.append({"run_id": "r1", "status": "completed"})
    store.append({"run_id": "r2", "status": "completed"})
    assert [row["run_id"] for row in store.all()] == ["r1", "r2"]


def test_finite_judge_actions() -> None:
    policy = JudgePolicy()
    action = policy.decide_exit_action(True, 0.9, "healthy")
    assert action in set(JudgeExitAction)


def test_runtime_event_typing() -> None:
    event = RuntimeEvent(
        event_id="e1",
        run_id="r1",
        step=1,
        category=RuntimeEventCategory.METRIC,
        message="ok",
    )
    assert event.category.value == "metric"


def test_stage_aware_interpretation() -> None:
    strict_conf, strict_issue = interpret_stage("strict", deterministic_passed=False)
    lenient_conf, lenient_issue = interpret_stage("lenient", deterministic_passed=False)
    assert strict_conf < lenient_conf
    assert strict_issue == "deterministic_regression"
    assert lenient_issue == "instability"


def test_deterministic_regression_blocks_promotion() -> None:
    policy = JudgePolicy()
    action = policy.decide_exit_action(False, 0.95, "healthy")
    assert action == JudgeExitAction.REJECT_CHECKPOINT


def test_control_spine_order_and_end_to_end_loop(tmp_path: Path) -> None:
    orch = build_orchestrator(state_root=tmp_path, run_id="run-001")
    results = orch.run("run-001")
    assert [result.phase for result in results] == list(SPINE_ORDER)

    lineage = LineageStore(tmp_path).get_current()
    runs = RunStore(tmp_path).all()
    decisions = DecisionStore(tmp_path).all()

    assert lineage is not None
    assert lineage["latest_run_id"] == "run-001"
    assert len(runs) == 1
    assert runs[0]["phase_order"] == [phase.value for phase in SPINE_ORDER]
    assert {decision["role"] for decision in decisions} == {"judge_entry", "judge_exit"}


def test_role_boundary_preservation(tmp_path: Path) -> None:
    orch = build_orchestrator(state_root=tmp_path, run_id="run-002")
    results = orch.run("run-002")
    outputs = {result.phase.value: result.output for result in results}
    assert "judge_exit" in outputs
    assert outputs["judge_exit"]["next_action"] in [action.value for action in JudgeExitAction]
    assert "outcome" in outputs["runtime_monitor"]


class VariantBackend(DryRunBackend):
    def __init__(self, runtime_variant: str) -> None:
        self.runtime_variant = runtime_variant

    def runtime_events(self, run_id: str) -> list[RuntimeEvent]:
        base = super().runtime_events(run_id)
        if self.runtime_variant == "soft_suspicion":
            return [
                *base,
                RuntimeEvent(
                    event_id=f"{run_id}-inc-1",
                    run_id=run_id,
                    step=220,
                    category=RuntimeEventCategory.INCIDENT,
                    message="soft incident",
                    payload_ref=f"artifacts/{run_id}/incident_1.json",
                ),
            ]
        if self.runtime_variant == "waste_stop":
            return [
                *base,
                RuntimeEvent(event_id=f"{run_id}-inc-1", run_id=run_id, step=220, category=RuntimeEventCategory.INCIDENT, message="incident 1", payload_ref=f"artifacts/{run_id}/incident_1.json"),
                RuntimeEvent(event_id=f"{run_id}-inc-2", run_id=run_id, step=221, category=RuntimeEventCategory.INCIDENT, message="incident 2", payload_ref=f"artifacts/{run_id}/incident_2.json"),
                RuntimeEvent(event_id=f"{run_id}-inc-3", run_id=run_id, step=222, category=RuntimeEventCategory.INCIDENT, message="incident 3", payload_ref=f"artifacts/{run_id}/incident_3.json"),
            ]
        if self.runtime_variant == "hard_abort":
            return [
                RuntimeEvent(
                    event_id=f"{run_id}-det-fail",
                    run_id=run_id,
                    step=120,
                    category=RuntimeEventCategory.DETERMINISTIC_CHECK,
                    message="deterministic checks fail",
                    payload_ref=f"artifacts/{run_id}/det_fail.json",
                )
            ]
        return base


@pytest.mark.parametrize(
    ("runtime_variant", "expected_outcome", "expected_action"),
    [
        ("soft_suspicion", "soft_suspicion", JudgeExitAction.PROMOTE_CHECKPOINT.value),
        ("waste_stop", "waste_stop", JudgeExitAction.RERUN_SAME_CONFIG.value),
        ("hard_abort", "hard_abort", JudgeExitAction.ABORT_RUN.value),
    ],
)
def test_adverse_runtime_paths(runtime_variant: str, expected_outcome: str, expected_action: str, tmp_path: Path) -> None:
    run_id = f"run-{runtime_variant}"
    orch = build_orchestrator(
        state_root=tmp_path,
        run_id=run_id,
        backend=VariantBackend(runtime_variant),
    )
    results = orch.run(run_id)
    outputs = {result.phase.value: result.output for result in results}

    assert outputs["runtime_monitor"]["outcome"] == expected_outcome
    assert outputs["judge_exit"]["next_action"] == expected_action
    assert outputs["judge_exit"]["next_action"] in [action.value for action in JudgeExitAction]
