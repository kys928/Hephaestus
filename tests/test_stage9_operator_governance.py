from __future__ import annotations

import json
import shutil
from pathlib import Path

from hephaestus.backends.ardor.backend import ArdorBackend
from hephaestus.backends.dry_run_backend import DryRunBackend
from hephaestus.backends.local_process_backend import LocalProcessBackend
from hephaestus.control.orchestrator import build_orchestrator
from hephaestus.policy.approval_policy import ApprovalPolicy
from hephaestus.policy.judge_policy import JudgePolicy
from hephaestus.policy.stage_policy import StagePolicy
from hephaestus.schemas.judge_exit import JudgeExitAction
from hephaestus.schemas.stage_profile import StageProfile
from hephaestus.state.decision_store import DecisionStore
from hephaestus.state.lineage_store import LineageStore
from hephaestus.state.query import Query


_FIXTURE_RUNNER = "tests/fixtures/fake_ardor_runner.py"


class ForceActionPolicy(JudgePolicy):
    def __init__(self, action: JudgeExitAction) -> None:
        super().__init__()
        self._action = action

    def decide_exit_action(self, *args, **kwargs) -> JudgeExitAction:  # type: ignore[override]
        return self._action


class AllowOnlyActionStagePolicy(StagePolicy):
    def __init__(self, action: JudgeExitAction) -> None:
        super().__init__()
        self._action = action

    def resolve(self, stage_name: str) -> StageProfile:  # type: ignore[override]
        return StageProfile(
            name=stage_name,
            strictness="strict",
            eval_pack="generic_lm",
            deterministic_gates={"max_toxicity": 0.4, "min_probe_score": 0.1},
            allowed_next_actions=[self._action.value],
            certification_profile={"eligibility": "bounded", "require_recheck": False, "min_consistent_runs": 1},
        )


def _config_dir_with_ardor_runner(tmp_path: Path) -> Path:
    config_dir = tmp_path / "configs-with-runner"
    shutil.copytree("configs", config_dir)
    ardor_cfg = json.loads((config_dir / "backends" / "ardor.yaml").read_text())
    ardor_cfg["local_runner_path"] = _FIXTURE_RUNNER
    (config_dir / "backends" / "ardor.yaml").write_text(json.dumps(ardor_cfg, indent=2))
    return config_dir


def test_stage9_low_risk_action_auto_allowed(tmp_path: Path) -> None:
    orch = build_orchestrator(
        state_root=tmp_path,
        run_id="s9-auto",
        judge_policy=ForceActionPolicy(JudgeExitAction.CONTINUE_LINEAGE_BEST),
    )
    orch.run("s9-auto")
    decisions = DecisionStore(tmp_path)
    assert decisions.all_approval_requests() == []
    lineage = LineageStore(tmp_path).get_current("lineage-main")
    assert lineage is not None
    assert lineage["last_approval_status"] == "not_required"


def test_stage9_high_impact_promotion_requires_approval_and_does_not_auto_execute(tmp_path: Path) -> None:
    orch = build_orchestrator(
        state_root=tmp_path,
        run_id="s9-promote-pending",
        stage_name="stabilization",
        judge_policy=ForceActionPolicy(JudgeExitAction.PROMOTE_CHECKPOINT),
    )
    orch.run("s9-promote-pending")

    decisions = DecisionStore(tmp_path)
    requests = decisions.all_approval_requests()
    assert len(requests) == 1
    assert requests[0]["proposed_action"] == "promote_checkpoint"
    assert requests[0]["status"] == "pending"

    lineage = LineageStore(tmp_path).get_current("lineage-main")
    assert lineage is not None
    assert lineage["pending_approval"] is True
    assert lineage["last_effective_action"] == "continue_lineage_best"


def test_stage9_approved_promotion_executes_and_updates_lineage_truth(tmp_path: Path) -> None:
    orch = build_orchestrator(
        state_root=tmp_path,
        run_id="s9-promote-approved",
        stage_name="stabilization",
        judge_policy=ForceActionPolicy(JudgeExitAction.PROMOTE_CHECKPOINT),
        operator_responses={
            "promote_checkpoint": {
                "outcome": "approved",
                "operator_id": "operator.alice",
                "note": "approved for bounded rollout",
            }
        },
    )
    orch.run("s9-promote-approved")
    lineage = LineageStore(tmp_path).get_current("lineage-main")
    assert lineage is not None
    assert lineage["pending_approval"] is False
    assert lineage["last_approval_status"] == "approved"
    assert lineage["last_effective_action"] == "promote_checkpoint"


def test_stage9_rejected_promotion_leaves_lineage_truth_unchanged_and_records_decision(tmp_path: Path) -> None:
    seed = LineageStore(tmp_path)
    seed.set_current(
        {
            "lineage_id": "lineage-main",
            "parent_lineage_id": None,
            "stage_name": "stabilization",
            "status": "active",
            "trust_level": "high",
            "loop_index": 1,
            "latest_run_id": "seed-run",
            "best_checkpoint_ref": "artifacts/seed/best.ckpt",
            "last_stable_checkpoint_ref": "artifacts/seed/stable.ckpt",
            "certified_stable_checkpoint_ref": "artifacts/seed/stable.ckpt",
            "last_certification_result": "certification_passed",
            "recent_failures": [],
            "known_pathologies": [],
            "last_decision": "promote_checkpoint",
            "last_decision_id": "dec-seed-exit",
            "branch_origin_checkpoint_ref": None,
            "child_lineage_ids": [],
            "run_count": 1,
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
    )
    orch = build_orchestrator(
        state_root=tmp_path,
        run_id="s9-promote-rejected",
        stage_name="stabilization",
        judge_policy=ForceActionPolicy(JudgeExitAction.PROMOTE_CHECKPOINT),
        operator_responses={
            "promote_checkpoint": {
                "outcome": "rejected",
                "operator_id": "operator.bob",
                "note": "insufficient governance confidence",
            }
        },
    )
    orch.run("s9-promote-rejected")
    lineage = LineageStore(tmp_path).get_current("lineage-main")
    assert lineage is not None
    assert lineage["best_checkpoint_ref"] == "artifacts/seed/best.ckpt"
    assert lineage["last_stable_checkpoint_ref"] == "artifacts/seed/stable.ckpt"
    assert lineage["last_approval_status"] == "rejected"

    latest_approval = Query(tmp_path).latest_approval_decision_for_lineage("lineage-main")
    assert latest_approval is not None
    assert latest_approval["status"] == "rejected"
    assert lineage["recent_failures"] == []
    assert lineage["known_pathologies"] == []


def test_stage9_rollback_restart_branch_are_gated(tmp_path: Path) -> None:
    for action in [
        JudgeExitAction.ROLLBACK_TO_CHECKPOINT,
        JudgeExitAction.RESTART_LINEAGE,
        JudgeExitAction.BRANCH_NEW_EXPERIMENT,
    ]:
        state_root = tmp_path / action.value
        orch = build_orchestrator(
            state_root=state_root,
            run_id=f"s9-{action.value}",
            stage_name="stabilization",
            judge_policy=ForceActionPolicy(action),
            stage_policy=AllowOnlyActionStagePolicy(action),
        )
        orch.run(f"s9-{action.value}")
        pending = Query(state_root).pending_approvals("lineage-main")
        assert len(pending) == 1
        assert pending[0]["proposed_action"] == action.value


def test_stage9_pending_approval_queries_are_supported(tmp_path: Path) -> None:
    orch = build_orchestrator(
        state_root=tmp_path,
        run_id="s9-query-pending",
        stage_name="stabilization",
        judge_policy=ForceActionPolicy(JudgeExitAction.PROMOTE_CHECKPOINT),
    )
    orch.run("s9-query-pending")
    query = Query(tmp_path)
    pending = query.pending_approvals("lineage-main")
    assert len(pending) == 1
    assert pending[0]["status"] == "pending"


def test_stage9_override_is_explicit_and_original_judgment_is_preserved(tmp_path: Path) -> None:
    orch = build_orchestrator(
        state_root=tmp_path,
        run_id="s9-override",
        stage_name="stabilization",
        judge_policy=ForceActionPolicy(JudgeExitAction.PROMOTE_CHECKPOINT),
        operator_responses={
            "promote_checkpoint": {
                "outcome": "override_approved",
                "operator_id": "operator.carol",
                "note": "manual emergency override",
            }
        },
    )
    orch.run("s9-override")
    decision_store = DecisionStore(tmp_path)
    judge_decision = decision_store.get("dec-s9-override-exit")
    approvals = decision_store.all_approval_decisions()
    assert judge_decision is not None
    assert judge_decision["action"] == "promote_checkpoint"
    assert judge_decision["metadata"]["operator_override"] is True
    assert approvals[-1]["outcome"] == "override_approved"

    lineage = LineageStore(tmp_path).get_current("lineage-main")
    assert lineage is not None
    assert lineage["last_requested_action"] == "promote_checkpoint"
    assert lineage["last_effective_action"] == "promote_checkpoint"


def test_stage9_override_not_allowed_blocks_override_and_is_auditable(tmp_path: Path) -> None:
    orch = build_orchestrator(
        state_root=tmp_path,
        run_id="s9-no-override",
        stage_name="stabilization",
        judge_policy=ForceActionPolicy(JudgeExitAction.PROMOTE_CHECKPOINT),
        approval_policy=ApprovalPolicy(
            action_rules={"promote_checkpoint": "override_not_allowed"},
            high_risk_actions={"promote_checkpoint"},
        ),
        operator_responses={
            "promote_checkpoint": {
                "outcome": "override_approved",
                "operator_id": "operator.denied",
                "note": "attempted override",
            }
        },
    )
    orch.run("s9-no-override")
    decision_store = DecisionStore(tmp_path)
    judge_decision = decision_store.get("dec-s9-no-override-exit")
    approval = decision_store.all_approval_decisions()[-1]
    lineage = LineageStore(tmp_path).get_current("lineage-main")

    assert judge_decision is not None
    assert judge_decision["action"] == "promote_checkpoint"
    assert judge_decision["metadata"]["override_blocked"] is True
    assert judge_decision["metadata"]["operator_override"] is False
    assert approval["status"] == "rejected"
    assert approval["metadata"]["override_blocked"] is True
    assert approval["metadata"]["resolution_reason"] == "override_not_allowed_by_policy"
    assert lineage is not None
    assert lineage["last_effective_action"] == "continue_lineage_best"
    assert lineage["last_requested_action"] == "promote_checkpoint"


def test_stage9_dry_local_ardor_paths_still_work_under_governance_gating(tmp_path: Path) -> None:
    dry = build_orchestrator(
        state_root=tmp_path / "dry",
        run_id="s9-dry",
        stage_name="stabilization",
        backend=DryRunBackend(),
        judge_policy=ForceActionPolicy(JudgeExitAction.PROMOTE_CHECKPOINT),
    )
    local = build_orchestrator(
        state_root=tmp_path / "local",
        run_id="s9-local",
        stage_name="stabilization",
        backend=LocalProcessBackend(),
        judge_policy=ForceActionPolicy(JudgeExitAction.PROMOTE_CHECKPOINT),
    )
    ardor = build_orchestrator(
        state_root=tmp_path / "ardor",
        run_id="s9-ardor",
        stage_name="stabilization",
        backend=ArdorBackend(config_dir=_config_dir_with_ardor_runner(tmp_path)),
        judge_policy=ForceActionPolicy(JudgeExitAction.PROMOTE_CHECKPOINT),
    )
    for run_id in ["s9-dry", "s9-local", "s9-ardor"]:
        root = Path("artifacts") / run_id
        root.mkdir(parents=True, exist_ok=True)
        (root / "processed_dataset.jsonl").write_text('{"text":"sample"}\n')

    dry.run("s9-dry")
    local.run("s9-local")
    ardor.run("s9-ardor")

    assert Query(tmp_path / "dry").pending_approvals("lineage-main")
    assert Query(tmp_path / "local").pending_approvals("lineage-main")
    assert Query(tmp_path / "ardor").pending_approvals("lineage-main")
