"""Integration audit #151: auto-allowed actions must not corrupt lineage checkpoint truth."""
from pathlib import Path

import pytest

from approval_fixtures import synthetic_checkpoint
from hephaestus.production.actions import GovernedActionExecutor
from hephaestus.production.state import ACTION_EXECUTIONS
from hephaestus.schemas.lineage_state import LineageState
from hephaestus.state.lineage_store import LineageStore
from hephaestus.storage.sqlite import SQLiteStateRepository


def _executor(root: Path, *, status: str, best: str) -> GovernedActionExecutor:
    LineageStore(root).set_current(LineageState(
        lineage_id="lineage-a", stage_name="smoke_test",
        status=status, best_checkpoint_ref=best,
    ).to_dict())
    return GovernedActionExecutor(root, SQLiteStateRepository(root / "state.sqlite3"))


def test_automatic_continuation_cannot_replace_best_with_foreign_checkpoint(tmp_path):
    valid = synthetic_checkpoint(tmp_path, "lineage-a-valid")
    foreign = synthetic_checkpoint(tmp_path, "lineage-b-valid")
    root = tmp_path / "state"
    executor = _executor(root, status="active", best=valid)
    with pytest.raises(PermissionError, match="does not belong to lineage"):
        executor.apply(lineage_id="lineage-a", run_id="foreign-run",
                       requested_action="continue_from_checkpoint", checkpoint_ref=foreign)
    assert LineageStore(root).get_current("lineage-a")["best_checkpoint_ref"] == valid
    assert executor.repository.all(ACTION_EXECUTIONS) == []


def test_automatic_continuation_rejects_tampered_known_checkpoint(tmp_path):
    valid = synthetic_checkpoint(tmp_path, "tampered")
    root = tmp_path / "state"
    executor = _executor(root, status="active", best=valid)
    (Path(valid) / "checkpoint_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(PermissionError, match="manifest is invalid"):
        executor.apply(lineage_id="lineage-a", run_id="tampered-run",
                       requested_action="continue_from_checkpoint", checkpoint_ref=valid)
    assert executor.repository.all(ACTION_EXECUTIONS) == []


@pytest.mark.parametrize("status", ["poisoned", "archived", "deprecated", "blocked"])
@pytest.mark.parametrize("action", ["continue_lineage_best", "continue_from_checkpoint",
                                    "rerun_same_config", "reject_candidate"])
def test_retired_or_poisoned_lineage_cannot_reactivate_via_auto_action(tmp_path,status,action):
    root = tmp_path / "state"
    executor = _executor(root, status=status, best="checkpoint://old")
    with pytest.raises(PermissionError, match="cannot autonomously continue"):
        executor.apply(lineage_id="lineage-a", run_id="forbidden-"+action,
                       requested_action=action, checkpoint_ref=None)
    assert LineageStore(root).get_current("lineage-a")["status"] == status
    assert executor.repository.all(ACTION_EXECUTIONS) == []


def test_verified_lineage_owned_checkpoint_continuation_remains_possible(tmp_path):
    valid = synthetic_checkpoint(tmp_path, "valid")
    root = tmp_path / "state"
    executor = _executor(root, status="active", best=valid)
    result = executor.apply(lineage_id="lineage-a", run_id="valid-run",
                            requested_action="continue_from_checkpoint", checkpoint_ref=valid)
    assert result["status"] == "applied"
    assert result["after"]["best_checkpoint_ref"] == valid


def test_actual_abort_cannot_be_undone_by_automatic_continuation(tmp_path):
    root = tmp_path / "state"
    executor = _executor(root, status="active", best="checkpoint://old")
    aborted = executor.apply(lineage_id="lineage-a", run_id="abort-one",
                             requested_action="abort_run")
    assert aborted["after"]["status"] == "blocked"
    with pytest.raises(PermissionError, match="cannot autonomously continue"):
        executor.apply(lineage_id="lineage-a", run_id="next-cycle",
                       requested_action="continue_lineage_best")
    assert LineageStore(root).get_current("lineage-a")["status"] == "blocked"
    assert len([r for r in executor.repository.all(ACTION_EXECUTIONS)
                if r["status"] == "applied"]) == 1
