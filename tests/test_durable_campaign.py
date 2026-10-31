"""Persistence tests only; no training, role inference or stage-count claims."""
from dataclasses import replace
import pytest

from hephaestus.production.durable_campaign import (
    BOOTSTRAP_COLLECTION, create_fresh_campaign_workspace,
    verify_campaign_workspace,
)
from hephaestus.storage.sqlite import SQLiteStateRepository

IDENTITY = {
    "workflow_id": "fresh-run-first-real-campaign",
    "run_id": "new-real-run-001",
    "lineage_id": "new-real-lineage",
    "stage_name": "smoke_test",
}


def fresh(tmp_path):
    return create_fresh_campaign_workspace(tmp_path / "campaign-001", **IDENTITY)


def test_persisted_new_state_has_zero_real_operations(tmp_path):
    workspace = fresh(tmp_path)
    assert workspace.state_path.is_file()
    report = verify_campaign_workspace(workspace)
    assert report.verified_completed_prefix == 0
    assert report.authentic_persisted_receipts == 0
    assert report.next_operation == "judge_entry.judge_entry_decision"
    assert not report.completed_campaign
    assert not report.dataset_training_approval_conferred
    assert not report.certified_role_execution_proven


def test_bootstrap_survives_separate_reopen_without_state_mutation(tmp_path):
    workspace = fresh(tmp_path)
    repo = SQLiteStateRepository(workspace.state_path, initialize=False)
    before = repo.all(BOOTSTRAP_COLLECTION)
    assert len(before) == 1
    assert verify_campaign_workspace(workspace).verified_completed_prefix == 0
    assert verify_campaign_workspace(workspace).verified_completed_prefix == 0
    assert repo.all(BOOTSTRAP_COLLECTION) == before


def test_original_path_is_never_reused(tmp_path):
    workspace = fresh(tmp_path)
    with pytest.raises(FileExistsError):
        create_fresh_campaign_workspace(workspace.root, **IDENTITY)
    (tmp_path / "occupied").mkdir()
    with pytest.raises(FileExistsError):
        create_fresh_campaign_workspace(tmp_path / "occupied", **IDENTITY)


def test_changed_identity_and_missing_original_db_fail(tmp_path):
    workspace = fresh(tmp_path)
    with pytest.raises(ValueError, match="identity/reference"):
        verify_campaign_workspace(replace(workspace, run_id="wrong-run"))
    with pytest.raises(ValueError, match="identity/reference"):
        verify_campaign_workspace(replace(
            workspace, bootstrap_artifact_ref="sha256:" + "0" * 64))
    with pytest.raises(FileNotFoundError, match="SQLite"):
        verify_campaign_workspace(replace(workspace, root=tmp_path / "absent"))


def test_invalid_identity_has_no_side_effect(tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        create_fresh_campaign_workspace("relative-path", **IDENTITY)
    with pytest.raises(ValueError, match="identity"):
        create_fresh_campaign_workspace(
            tmp_path / "invalid", **{**IDENTITY, "run_id": " "})
    assert not (tmp_path / "invalid").exists()
