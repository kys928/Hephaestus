from __future__ import annotations

from types import SimpleNamespace

from hephaestus.control.orchestrator import build_orchestrator
from hephaestus.policy.action_registry import evaluate_action_boundary
from hephaestus.state.decision_store import DecisionStore


def test_known_auto_action_boundary() -> None:
    result = evaluate_action_boundary("continue_lineage_best")
    assert result["known_action"] is True
    assert result["allowed"] is True
    assert result["requires_approval"] is False
    assert result["forbidden"] is False


def test_approval_required_action_boundary_is_explicit() -> None:
    result = evaluate_action_boundary("branch_new_experiment")
    assert result["known_action"] is True
    assert result["requires_approval"] is True
    assert result["forbidden"] is False
    assert result["category"] == "approval_required"


def test_high_risk_action_requires_approval() -> None:
    result = evaluate_action_boundary("promote_checkpoint")
    assert result["known_action"] is True
    assert result["high_risk"] is True
    assert result["requires_approval"] is True


def test_forbidden_action_is_blocked_even_with_approval_context() -> None:
    result = evaluate_action_boundary("mutate_frozen_eval_pack", context={"approval_status": "approved"})
    assert result["known_action"] is True
    assert result["forbidden"] is True
    assert result["allowed"] is False


def test_unknown_action_is_not_auto_allowed() -> None:
    result = evaluate_action_boundary("invent_new_autonomy_action")
    assert result["known_action"] is False
    assert result["allowed"] is False
    assert result["requires_approval"] is True


def test_decision_metadata_includes_action_boundary(tmp_path) -> None:
    orch = build_orchestrator(
        state_root=tmp_path,
        run_id="abp-normal",
    )
    orch.run("abp-normal")

    decision = DecisionStore(tmp_path).get("dec-abp-normal-exit")
    assert decision is not None
    metadata = decision["metadata"]
    assert "action_boundary" in metadata
    assert "action_category" in metadata
    assert "action_forbidden" in metadata
    assert "action_requires_approval" in metadata
    assert "action_high_risk" in metadata


def test_forbidden_action_cannot_become_effective_action(tmp_path, monkeypatch) -> None:
    from hephaestus.roles import judge_exit as judge_exit_module

    original_run = judge_exit_module.JudgeExitRole.run

    def _fake_run(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            run_id="abp-forbidden",
            lineage_id="lineage-main",
            verdict="blocked",
            next_action=SimpleNamespace(value="mutate_frozen_eval_pack"),
            confidence=0.0,
            reasons=["forced_forbidden_action_test"],
            to_dict=lambda: {
                "run_id": "abp-forbidden",
                "lineage_id": "lineage-main",
                "verdict": "blocked",
                "next_action": "mutate_frozen_eval_pack",
                "confidence": 0.0,
                "reasons": ["forced_forbidden_action_test"],
            },
        )

    monkeypatch.setattr(judge_exit_module.JudgeExitRole, "run", _fake_run)
    try:
        orch = build_orchestrator(state_root=tmp_path, run_id="abp-forbidden")
        orch.run("abp-forbidden")
    finally:
        monkeypatch.setattr(judge_exit_module.JudgeExitRole, "run", original_run)

    decision = DecisionStore(tmp_path).get("dec-abp-forbidden-exit")
    assert decision is not None
    metadata = decision["metadata"]
    assert metadata["action_forbidden"] is True
    assert metadata["effective_action"] == "continue_lineage_best"
    assert metadata["requested_action"] == "mutate_frozen_eval_pack"
