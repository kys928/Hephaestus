"""Audit F01/F02: retained producer identity and crash-boundary regressions."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from approval_fixtures import synthetic_checkpoint
from hephaestus.production.actions import GovernedActionExecutor
from hephaestus.production.state import ACTION_EXECUTIONS
from hephaestus.schemas.lineage_state import LineageState
from hephaestus.state.lineage_store import LineageStore
from hephaestus.storage.sqlite import SQLiteStateRepository
from hephaestus.storage.verified import canonical_bytes
from hephaestus.schemas.checkpoint_identity import CheckpointIdentity
from hephaestus.state.checkpoint_identity_store import CheckpointIdentityStore


def setup(root, *, retained=True):
    checkpoint = Path(synthetic_checkpoint(root, "accepted"))
    store = LineageStore(root)
    store.set_current(LineageState(lineage_id="L", stage_name="smoke_test",
                                  status="active", best_checkpoint_ref=str(checkpoint)).to_dict())
    repo = SQLiteStateRepository(root / "state.sqlite3")
    if retained:
        repo.append("checkpoint_identities", {
            "lineage_id": "L", "checkpoint_ref": str(checkpoint),
            "manifest_hash": json.loads((checkpoint / "checkpoint_manifest.json").read_text())["manifest_hash"],
            "evidence_ref": "synthetic:original-producer-receipt",
        })
    return GovernedActionExecutor(root, repo), checkpoint


def apply(executor, checkpoint):
    return executor.apply(lineage_id="L", run_id="R",
                          requested_action="continue_from_checkpoint", checkpoint_ref=str(checkpoint))


def test_coherently_rehashed_checkpoint_is_not_the_accepted_checkpoint(tmp_path):
    executor, checkpoint = setup(tmp_path)
    before = executor.lineage_store.get_current("L")
    (checkpoint / "model.safetensors").write_bytes(b"replacement-weights")
    components = {"model.safetensors": "sha256:" + hashlib.sha256(b"replacement-weights").hexdigest()}
    (checkpoint / "checkpoint_manifest.json").write_bytes(canonical_bytes({
        "components": components, "manifest_hash": "sha256:" + hashlib.sha256(canonical_bytes(components)).hexdigest(),
        "partial_write": False}))
    with pytest.raises(PermissionError, match="identity"):
        apply(executor, checkpoint)
    assert executor.lineage_store.get_current("L") == before
    assert executor.repository.all(ACTION_EXECUTIONS) == []


def test_legacy_checkpoint_requires_explicit_trusted_identity_migration(tmp_path):
    executor, checkpoint = setup(tmp_path, retained=False)
    with pytest.raises(PermissionError, match="identity"):
        apply(executor, checkpoint)
    assert executor.repository.all("checkpoint_identities") == []
    assert executor.repository.all(ACTION_EXECUTIONS) == []


def test_prepared_continuation_revalidates_before_mutation(tmp_path):
    executor, checkpoint = setup(tmp_path)
    before = executor.lineage_store.get_current("L")
    with patch.object(LineageStore, "set_current", side_effect=RuntimeError("crash-before-mutation")):
        with pytest.raises(RuntimeError, match="crash"):
            apply(executor, checkpoint)
    (checkpoint / "checkpoint_manifest.json").write_text("{}")
    with pytest.raises(PermissionError):
        executor.reconcile_prepared()
    assert executor.lineage_store.get_current("L") == before
    assert [r["status"] for r in executor.repository.all(ACTION_EXECUTIONS)] == ["prepared"]


def test_already_committed_after_state_only_finishes_journal(tmp_path):
    executor, checkpoint = setup(tmp_path)
    append = executor.repository.append

    def crash_on_applied(collection, record, **kwargs):
        if collection == ACTION_EXECUTIONS and record.get("status") == "applied":
            raise RuntimeError("crash-after-mutation")
        return append(collection, record, **kwargs)

    with patch.object(executor.repository, "append", side_effect=crash_on_applied):
        with pytest.raises(RuntimeError, match="crash"):
            apply(executor, checkpoint)
    after = executor.lineage_store.get_current("L")
    (checkpoint / "checkpoint_manifest.json").write_text("{}")
    assert executor.reconcile_prepared()[0]["status"] == "applied"
    assert executor.lineage_store.get_current("L") == after
    assert executor.reconcile_prepared() == []


def test_identity_store_never_replaces_an_accepted_commitment(tmp_path):
    executor, checkpoint = setup(tmp_path)
    with pytest.raises(PermissionError, match="replace"):
        CheckpointIdentityStore(executor.repository).retain(CheckpointIdentity(
            "L", str(checkpoint), "sha256:" + "f" * 64, "synthetic:new-receipt"))


def test_new_staged_checkpoint_membership_comes_from_its_lineage_bound_producer(tmp_path):
    executor, checkpoint = setup(tmp_path)
    current = executor.lineage_store.get_current("L")
    current["best_checkpoint_ref"] = None
    executor.lineage_store.set_current(current)
    assert apply(executor, checkpoint)["status"] == "applied"
    foreign = dict(executor.lineage_store.get_current("L"), lineage_id="foreign", best_checkpoint_ref=None)
    executor.lineage_store.set_current(foreign)
    with pytest.raises(PermissionError, match="belong"):
        executor.apply(lineage_id="foreign", run_id="foreign-run", requested_action="continue_from_checkpoint",
                       checkpoint_ref=str(checkpoint))


def test_resume_token_cannot_redefine_retained_checkpoint_identity(tmp_path):
    from types import SimpleNamespace
    from hephaestus.training.hf_lifecycle import TransformersTrainingLifecycleService
    executor, checkpoint = setup(tmp_path)
    original = executor.repository.all("checkpoint_identities")[0]["manifest_hash"]
    (checkpoint / "model.safetensors").write_bytes(b"replacement")
    components = {"model.safetensors": "sha256:" + hashlib.sha256(b"replacement").hexdigest()}
    new_hash = "sha256:" + hashlib.sha256(canonical_bytes(components)).hexdigest()
    (checkpoint / "checkpoint_manifest.json").write_bytes(canonical_bytes({
        "components": components, "manifest_hash": new_hash, "partial_write": False}))
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"config_fingerprint": "config"}))
    token = tmp_path / "resume.json"
    token.write_text(json.dumps({"checkpoint_ref": str(checkpoint),
        "checkpoint_manifest_hash": new_hash, "config_fingerprint": "config"}))
    managed = SimpleNamespace(job_ref=job,
        handle=SimpleNamespace(metadata={"checkpoint_manifest_hashes": {str(checkpoint): original}}))
    issue = TransformersTrainingLifecycleService._validate_resume(managed, token, {})
    assert issue.code == "checkpoint_identity_missing_or_changed"
