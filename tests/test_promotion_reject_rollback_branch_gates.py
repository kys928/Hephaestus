from __future__ import annotations

from pathlib import Path

from hephaestus.control.orchestrator import build_orchestrator
from hephaestus.policy.judge_policy import JudgePolicy
from hephaestus.policy.promotion_gates import evaluate_promotion_gates
from hephaestus.schemas.eval_report import EvalReport
from hephaestus.schemas.judge_exit import JudgeExitAction
from hephaestus.state.decision_store import DecisionStore
from hephaestus.state.lineage_store import LineageStore


class ForceActionPolicy(JudgePolicy):
    def __init__(self, action: JudgeExitAction) -> None:
        super().__init__()
        self._action = action

    def decide_exit_action(self, *args, **kwargs) -> JudgeExitAction:  # type: ignore[override]
        return self._action


def _report(
    *,
    deterministic_passed: bool = True,
    deterministic_scorecard: dict[str, object] | None = None,
    candidate: str | None = "artifacts/run/candidate.ckpt",
    eval_pack_integrity_level: str = "content_hash_verified",
    scorecard_integrity_level: str = "content_hash_verified",
    repeatability_sufficient: bool = True,
    variance_risk: str = "low",
) -> EvalReport:
    return EvalReport(
        eval_id="eval-1",
        run_id="run-1",
        stage_name="stabilization",
        pack_name="generic_lm",
        checkpoint_resolution={"selected_checkpoint_ref": candidate} if candidate else {},
        deterministic_scorecard=deterministic_scorecard if deterministic_scorecard is not None else {"deterministic_passed": deterministic_passed},
        deterministic_passed=deterministic_passed,
        eval_pack_integrity_level=eval_pack_integrity_level,
        scorecard_integrity_level=scorecard_integrity_level,
        repeatability_sufficient=repeatability_sufficient,
        variance_risk=variance_risk,
    )


def test_deterministic_failure_blocks_promotion_and_forces_safe_action() -> None:
    report = evaluate_promotion_gates(
        run_id="run-1",
        lineage_id="lineage-main",
        requested_action="promote_checkpoint",
        eval_report=_report(deterministic_passed=False),
        lineage_state={"status": "active", "best_checkpoint_ref": "artifacts/old/best.ckpt"},
        data_manifest={"manifest_integrity_level": "complete", "completeness_score": 1.0},
        approval_metadata={"approval_status": "approved"},
    )
    assert report.promotion_allowed is False
    assert report.blocking_failures
    assert report.recommended_effective_action in {"continue_lineage_best", "reject_checkpoint"}


def test_missing_scorecard_blocks_promotion_no_silent_pass() -> None:
    report = evaluate_promotion_gates(
        run_id="run-1",
        lineage_id="lineage-main",
        requested_action="promote_checkpoint",
        eval_report=_report(deterministic_passed=True, deterministic_scorecard={}),
        lineage_state={"status": "active"},
        data_manifest={"manifest_integrity_level": "complete", "completeness_score": 1.0},
        approval_metadata={"approval_status": "approved"},
    )
    assert report.promotion_allowed is False
    assert any("Deterministic evidence missing" in item for item in report.blocking_failures)


def test_insufficient_eval_pack_blocks_promotion() -> None:
    report = evaluate_promotion_gates(
        run_id="run-1",
        lineage_id="lineage-main",
        requested_action="promote_checkpoint",
        eval_report=_report(eval_pack_integrity_level="insufficient"),
        lineage_state={"status": "active"},
        data_manifest={"manifest_integrity_level": "complete", "completeness_score": 1.0},
        approval_metadata={"approval_status": "approved"},
    )
    assert report.promotion_allowed is False
    assert any("Eval pack integrity" in item for item in report.blocking_failures)


def test_reference_only_eval_and_partial_data_lower_confidence_without_hard_stop() -> None:
    report = evaluate_promotion_gates(
        run_id="run-1",
        lineage_id="lineage-main",
        requested_action="continue_lineage_best",
        eval_report=_report(eval_pack_integrity_level="reference_only", scorecard_integrity_level="inline_unhashed"),
        lineage_state={"status": "active"},
        data_manifest={"manifest_integrity_level": "partial", "completeness_score": 0.6},
        approval_metadata={"approval_status": "not_required"},
    )
    assert report.promotion_allowed is True
    assert report.confidence_ceiling < 1.0
    assert report.warnings


def test_missing_candidate_checkpoint_blocks_promotion() -> None:
    report = evaluate_promotion_gates(
        run_id="run-1",
        lineage_id="lineage-main",
        requested_action="promote_checkpoint",
        eval_report=_report(candidate=None),
        lineage_state={"status": "active"},
        data_manifest={"manifest_integrity_level": "complete", "completeness_score": 1.0},
        approval_metadata={"approval_status": "approved"},
    )
    assert report.promotion_allowed is False
    assert report.candidate_checkpoint_ref is None


def test_poisoned_lineage_blocks_promotion() -> None:
    report = evaluate_promotion_gates(
        run_id="run-1",
        lineage_id="lineage-main",
        requested_action="promote_checkpoint",
        eval_report=_report(),
        lineage_state={"status": "poisoned"},
        data_manifest={"manifest_integrity_level": "complete", "completeness_score": 1.0},
        approval_metadata={"approval_status": "approved"},
    )
    assert report.promotion_allowed is False
    assert report.recommended_effective_action in {"continue_lineage_best", "reject_checkpoint"}


def test_rollback_without_checkpoint_is_blocked_and_falls_back() -> None:
    report = evaluate_promotion_gates(
        run_id="run-1",
        lineage_id="lineage-main",
        requested_action="rollback_to_checkpoint",
        eval_report=_report(),
        lineage_state={"status": "active", "best_checkpoint_ref": None, "last_stable_checkpoint_ref": None},
        data_manifest={"manifest_integrity_level": "complete", "completeness_score": 1.0},
        approval_metadata={"approval_status": "approved"},
    )
    assert report.rollback_allowed is False
    assert report.recommended_effective_action in {"branch_new_experiment", "continue_lineage_best"}


def test_approval_pending_blocks_high_impact_action() -> None:
    report = evaluate_promotion_gates(
        run_id="run-1",
        lineage_id="lineage-main",
        requested_action="promote_checkpoint",
        eval_report=_report(),
        lineage_state={"status": "active"},
        data_manifest={"manifest_integrity_level": "complete", "completeness_score": 1.0},
        approval_metadata={"approval_status": "pending"},
    )
    assert report.recommended_effective_action in {"continue_lineage_best", "reject_checkpoint"}
    assert any("Approval status 'pending'" in item for item in report.blocking_failures)


def test_orchestrator_dry_run_smoke_persists_gate_report_and_effective_action(tmp_path: Path) -> None:
    seed = LineageStore(tmp_path)
    seed.set_current(
        {
            "lineage_id": "lineage-main",
            "stage_name": "stabilization",
            "status": "active",
            "trust_level": "high",
            "loop_index": 1,
            "latest_run_id": "seed-run",
            "best_checkpoint_ref": "artifacts/seed/best.ckpt",
            "last_stable_checkpoint_ref": "artifacts/seed/stable.ckpt",
            "certified_stable_checkpoint_ref": "artifacts/seed/stable.ckpt",
            "last_certification_result": "certification_passed",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "run_count": 1,
            "recent_failures": [],
            "known_pathologies": [],
            "child_lineage_ids": [],
            "branch_origin_checkpoint_ref": None,
        }
    )

    run_id = "gate-orch-1"
    root = Path("artifacts") / run_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "processed_dataset.jsonl").write_text('{"text":"sample"}\n')
    orch = build_orchestrator(
        state_root=tmp_path,
        run_id=run_id,
        stage_name="stabilization",
        judge_policy=ForceActionPolicy(JudgeExitAction.PROMOTE_CHECKPOINT),
    )
    orch.run(run_id)

    decision = DecisionStore(tmp_path).get(f"dec-{run_id}-exit")
    assert decision is not None
    metadata = decision["metadata"]
    assert "promotion_gate_report" in metadata
    assert "blocking_failures" in metadata
    assert "gate_warnings" in metadata
    assert "effective_action" in metadata
    lineage = LineageStore(tmp_path).get_current("lineage-main")
    assert lineage is not None
    if metadata["promotion_allowed"] is False:
        assert lineage["best_checkpoint_ref"] == "artifacts/seed/best.ckpt"
        assert lineage["last_stable_checkpoint_ref"] == "artifacts/seed/stable.ckpt"
