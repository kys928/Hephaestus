from __future__ import annotations

import json
import shutil
from pathlib import Path

from hephaestus.backends.ardor.backend import ArdorBackend
from hephaestus.backends.dry_run_backend import DryRunBackend
from hephaestus.backends.local_process_backend import LocalProcessBackend
from hephaestus.control.branching import create_branch_state
from hephaestus.control.orchestrator import build_orchestrator
from hephaestus.control.promotion import apply_promotion
from hephaestus.control.restart import create_restart_state
from hephaestus.policy.judge_policy import JudgePolicy
from hephaestus.policy.promotion_policy import PromotionPolicy
from hephaestus.schemas.judge_exit import JudgeExitAction
from hephaestus.state.lineage_store import LineageStore
from hephaestus.state.query import Query
from hephaestus.state.run_store import RunStore

_FIXTURE_RUNNER = "tests/fixtures/fake_ardor_runner.py"


class ForceRollbackPolicy(JudgePolicy):
    def decide_exit_action(self, *args, **kwargs) -> JudgeExitAction:  # type: ignore[override]
        return JudgeExitAction.ROLLBACK_TO_CHECKPOINT


def _config_dir_with_ardor_runner(tmp_path: Path) -> Path:
    config_dir = tmp_path / "configs-with-runner"
    shutil.copytree("configs", config_dir)
    ardor_cfg = json.loads((config_dir / "backends" / "ardor.yaml").read_text())
    ardor_cfg["local_runner_path"] = _FIXTURE_RUNNER
    (config_dir / "backends" / "ardor.yaml").write_text(json.dumps(ardor_cfg, indent=2))
    return config_dir


def test_stage6_multi_run_lineage_progression(tmp_path: Path) -> None:
    orch1 = build_orchestrator(state_root=tmp_path, run_id="s6-run-1")
    orch1.run("s6-run-1")
    orch2 = build_orchestrator(state_root=tmp_path, run_id="s6-run-2")
    orch2.run("s6-run-2")

    lineage = LineageStore(tmp_path).get_current("lineage-main")
    runs = RunStore(tmp_path).all()

    assert lineage is not None
    assert lineage["latest_run_id"] == "s6-run-2"
    assert lineage["loop_index"] == 2
    assert lineage["run_count"] == 2
    assert [row["run_id"] for row in runs] == ["s6-run-1", "s6-run-2"]


def test_stage6_best_vs_stable_checkpoint_conservative(tmp_path: Path) -> None:
    build_orchestrator(state_root=tmp_path, run_id="s6-promote").run("s6-promote")
    lineage = LineageStore(tmp_path).get_current("lineage-main")

    assert lineage is not None
    assert lineage["best_checkpoint_ref"]
    assert lineage["last_stable_checkpoint_ref"] in {None, lineage["best_checkpoint_ref"]}


class AlwaysFailBackend(DryRunBackend):
    def launch_training(self, prepared_job):  # type: ignore[override]
        result = super().launch_training(prepared_job)
        result.status = "failed"
        result.intermediate_eval = {}
        return result


def test_stage6_repeated_failures_degrade_trust_and_track_failures(tmp_path: Path) -> None:
    build_orchestrator(state_root=tmp_path, run_id="s6-fail-1", backend=AlwaysFailBackend()).run("s6-fail-1")
    build_orchestrator(state_root=tmp_path, run_id="s6-fail-2", backend=AlwaysFailBackend()).run("s6-fail-2")
    build_orchestrator(state_root=tmp_path, run_id="s6-fail-3", backend=AlwaysFailBackend()).run("s6-fail-3")

    lineage = LineageStore(tmp_path).get_current("lineage-main")
    assert lineage is not None
    assert lineage["trust_level"] == "low"
    assert len(lineage["recent_failures"]) == 3


def test_stage6_rollback_transition_succeeds_with_stable_target(tmp_path: Path) -> None:
    lineage_store = LineageStore(tmp_path)
    lineage_store.set_current(
        {
            "lineage_id": "lineage-main",
            "parent_lineage_id": None,
            "stage_name": "stabilization",
            "status": "active",
            "trust_level": "medium",
            "loop_index": 1,
            "latest_run_id": "seed-run",
            "best_checkpoint_ref": "artifacts/seed/newer.ckpt",
            "last_stable_checkpoint_ref": "artifacts/seed/stable.ckpt",
            "recent_failures": ["seed-fail-1", "seed-fail-2"],
            "known_pathologies": [],
            "last_decision": "continue_lineage_best",
            "last_decision_id": "dec-seed-exit",
            "branch_origin_checkpoint_ref": None,
            "child_lineage_ids": [],
            "run_count": 1,
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
    )

    orch = build_orchestrator(
        state_root=tmp_path,
        run_id="s6-rollback-success",
        stage_name="stabilization",
        judge_policy=ForceRollbackPolicy(),
    )
    orch.run("s6-rollback-success")

    lineage = lineage_store.get_current("lineage-main")
    assert lineage is not None
    assert lineage["last_decision"] == "rollback_to_checkpoint"
    assert lineage["best_checkpoint_ref"] == "artifacts/seed/stable.ckpt"


def test_stage6_rollback_transition_fails_honestly_without_target(tmp_path: Path) -> None:
    lineage_store = LineageStore(tmp_path)
    lineage_store.set_current(
        {
            "lineage_id": "lineage-main",
            "parent_lineage_id": None,
            "stage_name": "stabilization",
            "status": "active",
            "trust_level": "medium",
            "loop_index": 1,
            "latest_run_id": "seed-run",
            "best_checkpoint_ref": "artifacts/seed/newer.ckpt",
            "last_stable_checkpoint_ref": None,
            "recent_failures": ["seed-fail-1", "seed-fail-2"],
            "known_pathologies": [],
            "last_decision": "continue_lineage_best",
            "last_decision_id": "dec-seed-exit",
            "branch_origin_checkpoint_ref": None,
            "child_lineage_ids": [],
            "run_count": 1,
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
    )

    orch = build_orchestrator(
        state_root=tmp_path,
        run_id="s6-rollback-fail",
        stage_name="stabilization",
        judge_policy=ForceRollbackPolicy(),
    )
    orch.run("s6-rollback-fail")

    lineage = lineage_store.get_current("lineage-main")
    assert lineage is not None
    assert lineage["last_decision"] == "rollback_to_checkpoint"
    assert lineage["status"] == "blocked"
    assert "no_valid_rollback_target" in lineage["known_pathologies"]


def test_stage6_branch_and_restart_semantics_are_explicit(tmp_path: Path) -> None:
    lineage_store = LineageStore(tmp_path)
    parent = {
        "lineage_id": "lineage-main",
        "parent_lineage_id": None,
        "stage_name": "early_pretraining",
        "status": "active",
        "trust_level": "medium",
        "loop_index": 4,
        "latest_run_id": "run-4",
        "best_checkpoint_ref": "artifacts/run-4/best.ckpt",
        "last_stable_checkpoint_ref": "artifacts/run-4/stable.ckpt",
        "recent_failures": ["run-2"],
        "known_pathologies": [],
        "last_decision": "continue_lineage_best",
        "last_decision_id": "dec-run-4-exit",
        "branch_origin_checkpoint_ref": None,
        "child_lineage_ids": [],
        "run_count": 4,
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    lineage_store.set_current(parent)

    branch = create_branch_state(parent, "lineage-main-branch-5", "early_pretraining", parent["best_checkpoint_ref"], "2026-01-02T00:00:00+00:00")
    lineage_store.set_current(branch.child_state)
    lineage_store.add_child("lineage-main", branch.child_lineage_id)

    child = lineage_store.get_current("lineage-main-branch-5")
    assert child is not None
    assert child["parent_lineage_id"] == "lineage-main"
    assert child["branch_origin_checkpoint_ref"] == "artifacts/run-4/best.ckpt"

    restart = create_restart_state(parent, "lineage-main", "early_pretraining", "2026-01-03T00:00:00+00:00", "suspect_lineage")
    lineage_store.set_current(restart.reset_state)
    restarted = lineage_store.get_current("lineage-main")
    assert restarted is not None
    assert restarted["status"] == "restarted"
    assert "suspect_lineage" in restarted["known_pathologies"]


def test_stage6_query_support_and_append_only_runs(tmp_path: Path) -> None:
    build_orchestrator(state_root=tmp_path, run_id="s6-q-1").run("s6-q-1")
    build_orchestrator(state_root=tmp_path, run_id="s6-q-2").run("s6-q-2")

    query = Query(tmp_path)
    latest = query.latest_run_in_lineage("lineage-main")
    decisions = query.recent_decisions("lineage-main")

    assert latest is not None
    assert latest["run_id"] == "s6-q-2"
    assert len(decisions) >= 2
    assert query.best_checkpoint("lineage-main")


def test_stage6_dry_local_ardor_paths_still_work_with_lineage(tmp_path: Path) -> None:
    dry = build_orchestrator(state_root=tmp_path / "dry", run_id="s6-dry", backend=DryRunBackend())
    local = build_orchestrator(state_root=tmp_path / "local", run_id="s6-local", backend=LocalProcessBackend())
    ardor = build_orchestrator(
        state_root=tmp_path / "ardor",
        run_id="s6-ardor",
        backend=ArdorBackend(config_dir=_config_dir_with_ardor_runner(tmp_path)),
    )

    for run_id in ["s6-dry", "s6-local", "s6-ardor"]:
        root = Path("artifacts") / run_id
        root.mkdir(parents=True, exist_ok=True)
        (root / "processed_dataset.jsonl").write_text('{"text":"sample"}\n')

    dry.run("s6-dry")
    local.run("s6-local")
    ardor.run("s6-ardor")

    assert LineageStore(tmp_path / "dry").get_current("lineage-main") is not None
    assert LineageStore(tmp_path / "local").get_current("lineage-main") is not None
    assert LineageStore(tmp_path / "ardor").get_current("lineage-main") is not None


def test_stage6_deterministic_regression_blocks_promote_and_stable_advancement() -> None:
    policy = PromotionPolicy()
    state_before = {
        "best_checkpoint_ref": "artifacts/old/best.ckpt",
        "last_stable_checkpoint_ref": "artifacts/old/stable.ckpt",
    }
    promotion_state = policy.decide(deterministic_passed=False, confidence=0.99, has_candidate=True)
    transition = apply_promotion(
        lineage_state=state_before,
        candidate_checkpoint_ref="artifacts/new/candidate.ckpt",
        promotion_state=promotion_state,
        deterministic_passed=False,
        confidence=0.99,
        stable_confidence_threshold=policy.min_confidence_for_stable,
    )

    assert promotion_state == "rejected"
    assert transition.best_checkpoint_ref == "artifacts/old/best.ckpt"
    assert transition.last_stable_checkpoint_ref == "artifacts/old/stable.ckpt"
    assert "stable_checkpoint_updated" not in transition.notes


def test_stage6_lineage_state_stays_compact(tmp_path: Path) -> None:
    build_orchestrator(state_root=tmp_path, run_id="s6-compact-1").run("s6-compact-1")
    build_orchestrator(state_root=tmp_path, run_id="s6-compact-2").run("s6-compact-2")
    lineage = LineageStore(tmp_path).get_current("lineage-main")
    assert lineage is not None
    assert "run_records" not in lineage
    assert "decision_records" not in lineage
    assert isinstance(lineage.get("recent_failures"), list)
