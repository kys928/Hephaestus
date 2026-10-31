"""Regression tests: no fabricated stages, no forged approval, no CPU role bypass.

All workflow states created in this file are *test fixtures only*, not evidence
that the canonical original campaign progressed.
"""
import copy
from dataclasses import replace
import pytest

from hephaestus.control.spine import SPINE_ORDER
from hephaestus.control.staged_state import (
    PHASE_SUBSTEPS, REQUIRED_RECORD_KINDS, WORKFLOW_RECORD_COLLECTION, WORKFLOW_STATE_COLLECTION,
    StagedStepState, StagedWorkflowState, StagedAutonomousDependencies,
    StagedAutonomousServices,
)
from hephaestus.production.governed_campaign_admission import (
    inspect_governed_campaign, resume_real_governed_campaign,
)
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.sqlite import SQLiteStateRepository
from hephaestus.storage.verified import canonical_bytes


ORDER = [(phase.value, s) for phase in SPINE_ORDER for s in PHASE_SUBSTEPS[phase]]
IDENTITY = {
    "workflow_id": "fixture-workflow-approval-blocker",
    "run_id": "fixture-run",
    "lineage_id": "fixture-lineage",
    "stage_name": "smoke_test",
}


def _repo(tmp_path):
    return SQLiteStateRepository(tmp_path / "state.sqlite"), FileSystemArtifactStore(tmp_path / "objects")


def _record(repository, store, *, phase, substep, op, payload, kind="substep_result"):
    wrapped = {
        "record_id": f"fixture-record-{phase}-{substep}-{kind}-{payload['attempt']}",
        **IDENTITY, "phase": phase, "substep": substep,
        "operation_id": op, "sequence": 1, "kind": kind,
        "payload": payload,
    }
    ref = store.put_bytes(canonical_bytes(wrapped), media_type="application/json")
    repository.append(WORKFLOW_RECORD_COLLECTION, {**wrapped, "artifact_ref": ref.artifact_ref})


def _fixture(tmp_path, *, n=11, pending=True, omit_approval=False):
    assert 0 <= n < 45
    repo, store = _repo(tmp_path)
    state = StagedWorkflowState(**IDENTITY)
    for i, (phase, substep) in enumerate(ORDER[:n]):
        op = f"fixture-op-{i}"
        critical=REQUIRED_RECORD_KINDS.get(substep)
        step = StagedStepState(
            phase, substep, op, status="completed", attempt=1,
            completion_marker=True, resumable=False,
            output={"record_kinds":[critical] if critical else []},
        )
        state.steps[f"{phase}.{substep}"] = step
        if critical:
            _record(repo, store, phase=phase, substep=substep, op=op,
                    kind=critical, payload={"attempt":1,"fixture_science_record":True})
        _record(repo, store, phase=phase, substep=substep, op=op,
                payload={"status": "completed", "attempt": 1,
                         "resumable": False, "completion_marker": True,
                         "input_refs": [], "output_refs": [], "blocking_issues": []})
    phase, substep = ORDER[n]
    if pending:
        assert substep == "acquisition_approval_gate" if n == 11 else True
        op = f"fixture-op-{n}"
        request = {
            "request_id": "fixture-existing-request",
            "operation_id": op, "subject_ref": "dataset-selection-fixture",
            "workflow_id": IDENTITY["workflow_id"], "run_id": IDENTITY["run_id"],
            "phase": phase, "substep": substep,
            "requirements": ["dataset_selection_approval", "dataset_acquisition"],
        }
        step = StagedStepState(
            phase, substep, op, status="approval_pending",
            attempt=1, completion_marker=False, resumable=True,
            blocking_issues=["approval_pending"],
            approval_request=None if omit_approval else request,
        )
        state.steps[f"{phase}.{substep}"] = step
        state.blocking_issues = ["approval_pending"]
        state.status = "approval_pending"
        state.current_phase = phase
        state.current_substep = substep
        _record(repo, store, phase=phase, substep=substep, op=op,
                payload={"status": "approval_pending", "attempt": 1,
                         "resumable": True, "completion_marker": False,
                         "input_refs": [], "output_refs": [],
                         "blocking_issues": ["approval_pending"]})
    repo.append(WORKFLOW_STATE_COLLECTION, state.to_dict())
    return repo, store, state


def _inspect(repo, store):
    return inspect_governed_campaign(repo, store, **IDENTITY)


def test_exact_11_stages_and_original_12th_approval_are_reported_not_advanced(tmp_path):
    repo, store, _ = _fixture(tmp_path)
    before = repo.all(WORKFLOW_STATE_COLLECTION)
    report = _inspect(repo, store)
    assert len(ORDER) == 45
    assert report.verified_completed_prefix == 11
    assert report.authentic_persisted_receipts == 12
    assert report.next_operation == "data_acquisition_audit.acquisition_approval_gate"
    assert report.pending_approval_request["subject_ref"] == "dataset-selection-fixture"
    assert set(report.pending_approval_request["requirements"]) == {
        "dataset_selection_approval", "dataset_acquisition",
    }
    assert "approval_pending" in report.blocking_issues
    assert report.resumable and not report.completed_campaign
    assert not report.certified_role_execution_proven
    assert not report.dataset_training_approval_conferred
    assert repo.all(WORKFLOW_STATE_COLLECTION) == before  # Strictly read-only


def test_forged_completion_with_missing_real_receipt_fails(tmp_path):
    repo, store, state = _fixture(tmp_path)
    next_key = "data_acquisition_audit.acquisition_approval_gate"
    state.steps[next_key] = replace(state.steps[next_key], status="completed",
                                     completion_marker=True)
    repo.append(WORKFLOW_STATE_COLLECTION, state.to_dict())
    with pytest.raises(ValueError, match="uniquely verified"):
        _inspect(repo, store)


def test_missing_real_pending_approval_request_blocks(tmp_path):
    repo, store, _ = _fixture(tmp_path, omit_approval=True)
    with pytest.raises(ValueError, match="genuine persisted request"):
        _inspect(repo, store)


def test_stage_status_without_original_science_record_declaration_fails(tmp_path):
    repo, store, state = _fixture(tmp_path)
    state.steps["judge_entry.judge_entry_decision"].output["record_kinds"]=[]
    repo.append(WORKFLOW_STATE_COLLECTION, state.to_dict())
    with pytest.raises(ValueError, match="scientific stage lacks declared"):
        _inspect(repo, store)


def test_wrong_run_cannot_resume_another_workflow(tmp_path):
    repo, store, _ = _fixture(tmp_path)
    with pytest.raises(PermissionError, match="identity"):
        inspect_governed_campaign(repo, store,
            **{**IDENTITY, "run_id": "other-run"})
    with pytest.raises(FileNotFoundError, match="persisted"):
        inspect_governed_campaign(repo, store,
            **{**IDENTITY, "workflow_id": "invented-workflow"})


def test_no_content_addressed_receipts_cannot_claim_real_progress(tmp_path):
    repo, store, state = _fixture(tmp_path)
    first_phase, first_substep = ORDER[0]
    # A forged state row alone does not complete a stage: original state
    # requires the separately hashed substep receipt.
    repository_rows = repo.all(WORKFLOW_RECORD_COLLECTION)
    assert repository_rows
    new = copy.deepcopy(state)
    new.steps[f"{first_phase}.{first_substep}"].operation_id = "invented-identity"
    repo.append(WORKFLOW_STATE_COLLECTION, new.to_dict())
    with pytest.raises(ValueError, match="uniquely verified"):
        _inspect(repo, store)


def test_resume_denies_missing_real_scoped_approvals_and_roles(tmp_path):
    repo, store, _ = _fixture(tmp_path)
    deps = StagedAutonomousDependencies(
        state_repository=repo, artifact_store=store,
        services=StagedAutonomousServices(),
        approval_service=None,
    )
    with pytest.raises(PermissionError, match="scoped approval"):
        resume_real_governed_campaign(deps, **IDENTITY)
    class _NoGrant:
        def decision_for(self, request):
            return None
    deps.approval_service = _NoGrant()
    with pytest.raises(PermissionError, match="five certified"):
        resume_real_governed_campaign(deps, **IDENTITY)
    assert _inspect(repo, store).verified_completed_prefix == 11
