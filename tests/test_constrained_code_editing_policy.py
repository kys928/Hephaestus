from __future__ import annotations

from hephaestus.policy.action_registry import evaluate_action_boundary
from hephaestus.policy.code_edit_policy import evaluate_code_edit_proposal
from hephaestus.state.code_edit_proposal_store import CodeEditProposalStore


def test_docs_tests_only_proposal_is_low_risk() -> None:
    proposal = evaluate_code_edit_proposal(
        {
            "proposal_id": "ced-1",
            "requested_by": "operator",
            "purpose": "Improve docs and tests",
            "target_files": ["docs/foo.md", "tests/test_x.py"],
            "rollback_plan": "Revert commit",
            "test_plan": ["pytest tests/test_x.py -q"],
        }
    )
    assert proposal.risk_level == "low"
    assert proposal.status == "approval_required"
    assert "operator_approval" in proposal.required_approvals


def test_source_policy_proposal_is_high_or_medium_and_requires_approval() -> None:
    proposal = evaluate_code_edit_proposal(
        {
            "proposal_id": "ced-2",
            "requested_by": "operator",
            "purpose": "Adjust policy classification",
            "target_files": ["src/hephaestus/policy/foo.py"],
            "rollback_plan": "Revert commit",
            "test_plan": ["pytest -q"],
        }
    )
    assert proposal.risk_level in {"high", "medium"}
    assert proposal.status == "approval_required"
    assert proposal.status != "blocked"


def test_forbidden_path_is_blocked() -> None:
    proposal = evaluate_code_edit_proposal(
        {
            "proposal_id": "ced-3",
            "requested_by": "operator",
            "purpose": "Try forbidden path",
            "target_files": ["state/runs.jsonl", "artifacts/model.pt"],
            "rollback_plan": "N/A",
            "test_plan": [],
        }
    )
    assert proposal.status == "blocked"
    assert proposal.risk_level == "forbidden"
    assert "state/runs.jsonl" in proposal.forbidden_files_touched
    assert "not_approvable_forbidden_path" in proposal.required_approvals


def test_secret_filename_is_blocked() -> None:
    proposal = evaluate_code_edit_proposal(
        {
            "proposal_id": "ced-4",
            "requested_by": "operator",
            "purpose": "Try env edit",
            "target_files": [".env", "secrets/token.json"],
            "rollback_plan": "N/A",
            "test_plan": [],
        }
    )
    assert proposal.status == "blocked"
    assert proposal.risk_level == "forbidden"


def test_model_weight_extension_is_blocked() -> None:
    proposal = evaluate_code_edit_proposal(
        {
            "proposal_id": "ced-5",
            "requested_by": "operator",
            "purpose": "Try checkpoint edit",
            "target_files": ["checkpoints/best_model.pt"],
            "rollback_plan": "N/A",
            "test_plan": [],
        }
    )
    assert proposal.status == "blocked"
    assert proposal.risk_level == "forbidden"


def test_store_round_trip_normalizes_through_policy(tmp_path) -> None:
    store = CodeEditProposalStore(tmp_path)
    store.append(
        {
            "proposal_id": "ced-store-1",
            "run_id": "run-1",
            "lineage_id": "lineage-1",
            "requested_by": "operator",
            "purpose": "Safe docs update",
            "target_files": ["docs/policy.md"],
            "rollback_plan": "Revert commit",
            "test_plan": ["pytest tests/test_constrained_code_editing_policy.py -q"],
        }
    )
    store.append(
        {
            "proposal_id": "ced-store-1",
            "run_id": "run-1",
            "lineage_id": "lineage-1",
            "requested_by": "operator",
            "purpose": "Duplicate should dedupe",
            "target_files": ["docs/policy.md"],
            "rollback_plan": "Revert commit",
            "test_plan": [],
        }
    )

    record = store.get("ced-store-1")
    assert record is not None
    assert record["status"] == "approval_required"
    assert record["risk_level"] == "low"

    by_status = store.list_by_status("approval_required")
    assert len(by_status) == 1
    assert by_status[0]["proposal_id"] == "ced-store-1"
    assert len(store.list_for_run("run-1")) == 1
    assert len(store.list_for_lineage("lineage-1")) == 1


def test_action_registry_code_edit_boundaries() -> None:
    propose = evaluate_action_boundary("propose_code_edit")
    assert propose["known_action"] is True
    assert propose["requires_approval"] is True
    assert propose["category"] == "approval_required"

    execute = evaluate_action_boundary("execute_code_edit")
    assert execute["known_action"] is True
    assert execute["requires_approval"] is True
    assert execute["high_risk"] is True
    assert execute["category"] == "high_risk_approval_required"

    forbidden_unapproved = evaluate_action_boundary("execute_unapproved_code_edit")
    assert forbidden_unapproved["known_action"] is True
    assert forbidden_unapproved["forbidden"] is True

    forbidden_path = evaluate_action_boundary("edit_forbidden_path")
    assert forbidden_path["known_action"] is True
    assert forbidden_path["forbidden"] is True
