from __future__ import annotations

from hephaestus.control.code_edit_workflow import CodeEditProposalWorkflow
from hephaestus.state.code_edit_proposal_store import CodeEditProposalStore


def _workflow(tmp_path):
    return CodeEditProposalWorkflow(CodeEditProposalStore(tmp_path))


def test_safe_docs_tests_proposal_persists_as_approval_required(tmp_path) -> None:
    workflow = _workflow(tmp_path)
    proposal = workflow.create_proposal(
        run_id="run-docs",
        lineage_id="lineage-docs",
        requested_by="operator",
        purpose="Update documentation and tests",
        target_files=["docs/constrained_code_editing_policy.md", "tests/test_stage11_code_edit_workflow.py"],
        rollback_plan="Revert the proposal commit",
        test_plan=["pytest tests/test_stage11_code_edit_workflow.py -q"],
        metadata={"stage": 11},
    )

    stored = workflow.store.get(proposal.proposal_id)
    assert stored is not None
    assert stored["status"] == "approval_required"
    assert stored["risk_level"] == "low"
    assert stored["run_id"] == "run-docs"
    assert stored["lineage_id"] == "lineage-docs"


def test_source_proposal_persists_as_medium_or_high_approval_required(tmp_path) -> None:
    workflow = _workflow(tmp_path)
    proposal = workflow.create_proposal(
        run_id="run-src",
        lineage_id="lineage-src",
        requested_by="operator",
        purpose="Add governed workflow helper",
        target_files=["src/hephaestus/control/code_edit_workflow.py"],
        rollback_plan="Revert the helper commit",
        test_plan=["pytest -q"],
    )

    assert proposal.status == "approval_required"
    assert proposal.risk_level in {"medium", "high"}


def test_forbidden_proposal_persists_as_blocked(tmp_path) -> None:
    workflow = _workflow(tmp_path)
    proposal = workflow.create_proposal(
        run_id="run-forbidden",
        lineage_id="lineage-forbidden",
        requested_by="operator",
        purpose="Forbidden edit attempt",
        target_files=["state/runs.jsonl", "artifacts/model.pt"],
        rollback_plan="No mutation allowed",
        test_plan=[],
    )

    assert proposal.status == "blocked"
    assert proposal.risk_level == "forbidden"
    assert "state/runs.jsonl" in proposal.forbidden_files_touched


def test_blocked_proposal_cannot_be_approved(tmp_path) -> None:
    workflow = _workflow(tmp_path)
    blocked = workflow.create_proposal(
        run_id="run-blocked",
        lineage_id="lineage-blocked",
        requested_by="operator",
        purpose="Forbidden edit attempt",
        target_files=[".env"],
        rollback_plan="No mutation allowed",
        test_plan=[],
    )

    approved = workflow.approve_proposal(blocked.proposal_id, operator_id="operator", note="should not approve")

    assert approved.status == "blocked"
    assert workflow.store.get(blocked.proposal_id)["status"] == "blocked"  # type: ignore[index]


def test_approved_proposal_can_produce_dry_run_execution_record(tmp_path) -> None:
    workflow = _workflow(tmp_path)
    proposal = workflow.create_proposal(
        run_id="run-approve",
        lineage_id="lineage-approve",
        requested_by="operator",
        purpose="Update docs",
        target_files=["docs/constrained_code_editing_policy.md"],
        rollback_plan="Revert docs commit",
        test_plan=["pytest -q"],
    )
    approved = workflow.approve_proposal(proposal.proposal_id, operator_id="operator", note="approved for dry run")

    execution = workflow.execute_dry_run(approved.proposal_id, requested_by="operator")

    assert approved.status == "approved"
    assert execution.status == "dry_run_ready"
    assert execution.dry_run is True
    assert execution.reason == "approved_proposal_dry_run_only_no_files_mutated"
    assert execution.target_files == ["docs/constrained_code_editing_policy.md"]


def test_unapproved_proposal_execution_is_refused(tmp_path) -> None:
    workflow = _workflow(tmp_path)
    proposal = workflow.create_proposal(
        run_id="run-unapproved",
        lineage_id="lineage-unapproved",
        requested_by="operator",
        purpose="Update docs",
        target_files=["docs/constrained_code_editing_policy.md"],
        rollback_plan="Revert docs commit",
        test_plan=["pytest -q"],
    )

    execution = workflow.execute_dry_run(proposal.proposal_id, requested_by="operator")

    assert execution.status == "refused"
    assert execution.reason == "proposal_not_approved"
    assert execution.dry_run is True


def test_proposal_can_be_rejected(tmp_path) -> None:
    workflow = _workflow(tmp_path)
    proposal = workflow.create_proposal(
        run_id="run-reject",
        lineage_id="lineage-reject",
        requested_by="operator",
        purpose="Update docs",
        target_files=["docs/constrained_code_editing_policy.md"],
        rollback_plan="Revert docs commit",
        test_plan=["pytest -q"],
    )

    rejected = workflow.reject_proposal(proposal.proposal_id, operator_id="operator", note="not needed")

    assert rejected.status == "rejected"
    assert workflow.store.get(proposal.proposal_id)["status"] == "rejected"  # type: ignore[index]


def test_query_helpers_work(tmp_path) -> None:
    workflow = _workflow(tmp_path)
    pending = workflow.create_proposal(
        run_id="run-query",
        lineage_id="lineage-query",
        requested_by="operator",
        purpose="Update docs",
        target_files=["docs/constrained_code_editing_policy.md"],
        rollback_plan="Revert docs commit",
        test_plan=["pytest -q"],
    )
    blocked = workflow.create_proposal(
        run_id="run-query",
        lineage_id="lineage-query-blocked",
        requested_by="operator",
        purpose="Forbidden edit attempt",
        target_files=["secrets/token.json"],
        rollback_plan="No mutation allowed",
        test_plan=[],
    )

    assert [row["proposal_id"] for row in workflow.store.list_pending()] == [pending.proposal_id]
    assert [row["proposal_id"] for row in workflow.store.list_blocked()] == [blocked.proposal_id]
    assert {row["proposal_id"] for row in workflow.store.list_for_run("run-query")} == {
        pending.proposal_id,
        blocked.proposal_id,
    }
    assert [row["proposal_id"] for row in workflow.store.list_for_lineage("lineage-query")] == [pending.proposal_id]


def test_approved_proposal_can_record_real_execution(tmp_path) -> None:
    workflow = _workflow(tmp_path)
    proposal = workflow.create_proposal(
        run_id="run-execute",
        lineage_id="lineage-execute",
        requested_by="operator",
        purpose="Update docs",
        target_files=["docs/constrained_code_editing_policy.md"],
        rollback_plan="Revert docs commit",
        test_plan=["pytest -q"],
    )
    approved = workflow.approve_proposal(proposal.proposal_id, operator_id="operator", note="approved for execution")

    execution = workflow.execute_approved(
        approved.proposal_id,
        requested_by="operator",
        changed_files=["docs/constrained_code_editing_policy.md"],
        metadata={"executor": "unit-test"},
    )

    assert execution.status == "executed"
    assert execution.dry_run is False
    assert execution.rollback_plan == "Revert docs commit"
    assert execution.changed_files == ["docs/constrained_code_editing_policy.md"]
    assert workflow.store.get(approved.proposal_id)["status"] == "executed"  # type: ignore[index]
    stored_executions = workflow.store.list_executions_for_proposal(approved.proposal_id)
    assert stored_executions[-1]["status"] == "executed"
    assert stored_executions[-1]["changed_files"] == ["docs/constrained_code_editing_policy.md"]


def test_real_execution_refuses_unapproved_or_unauthorized_paths(tmp_path) -> None:
    workflow = _workflow(tmp_path)
    proposal = workflow.create_proposal(
        run_id="run-refuse",
        lineage_id="lineage-refuse",
        requested_by="operator",
        purpose="Update docs",
        target_files=["docs/constrained_code_editing_policy.md"],
        rollback_plan="Revert docs commit",
        test_plan=["pytest -q"],
    )

    unapproved = workflow.execute_approved(
        proposal.proposal_id,
        requested_by="operator",
        changed_files=["docs/constrained_code_editing_policy.md"],
    )
    assert unapproved.status == "refused"
    assert unapproved.reason == "proposal_not_approved"

    approved = workflow.approve_proposal(proposal.proposal_id, operator_id="operator", note="approved for execution")
    escaped = workflow.execute_approved(
        approved.proposal_id,
        requested_by="operator",
        changed_files=["../secrets/token.json"],
    )
    assert escaped.status == "refused"
    assert escaped.reason == "unauthorized_path_access"

    outside_scope = workflow.execute_approved(
        approved.proposal_id,
        requested_by="operator",
        changed_files=["tests/test_stage11_code_edit_workflow.py"],
    )
    assert outside_scope.status == "refused"
    assert outside_scope.reason == "unauthorized_path_access"
    assert outside_scope.metadata["unauthorized_paths"] == ["tests/test_stage11_code_edit_workflow.py"]
