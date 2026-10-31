"""Fresh, isolated custody for a NEW governed 45-operation campaign.

This creates zero scientific operations and grants no approvals. Production
execution must still use the native staged orchestrator and real services.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hephaestus.control.staged_state import WORKFLOW_STATE_COLLECTION, StagedWorkflowState
from hephaestus.production.governed_campaign_admission import inspect_governed_campaign
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.sqlite import SQLiteStateRepository
from hephaestus.storage.verified import canonical_bytes, read_verified_json

BOOTSTRAP_COLLECTION = "governed_campaign_bootstraps"
BOOTSTRAP_SCHEMA = "hephaestus-new-governed-campaign-bootstrap.v1"


@dataclass(frozen=True, slots=True)
class CampaignWorkspace:
    root: Path
    workflow_id: str
    run_id: str
    lineage_id: str
    stage_name: str
    bootstrap_artifact_ref: str

    @property
    def state_path(self) -> Path:
        return self.root / "state.sqlite"

    @property
    def artifacts_path(self) -> Path:
        return self.root / "artifacts"

    @property
    def identity(self) -> dict[str, str]:
        return {
            "workflow_id": self.workflow_id, "run_id": self.run_id,
            "lineage_id": self.lineage_id, "stage_name": self.stage_name,
        }


def _bootstrap_payload(identity: dict[str, str]) -> dict[str, object]:
    return {
        "schema": BOOTSTRAP_SCHEMA, **identity,
        "source": "fresh_new_workflow_not_historical_resume",
        "historical_operation_receipts_imported": 0,
        "completed_operations_at_creation": 0,
        "approval_conferred": False,
        "dataset_admitted": False,
        "certified_role_execution_proven": False,
    }


def create_fresh_campaign_workspace(
    root: Path, *, workflow_id: str, run_id: str,
    lineage_id: str, stage_name: str,
) -> CampaignWorkspace:
    """Reserve a never-used directory and persist an authentic EMPTY state.

    The operator must choose a private persistent mount, NOT an ephemeral CI
    tmpdir. Directory creation is exclusive. No model/training work happens.
    """
    target = Path(root)
    if not target.is_absolute():
        raise ValueError("campaign workspace must be an absolute durable-volume path")
    identity = dict(workflow_id=workflow_id, run_id=run_id,
                    lineage_id=lineage_id, stage_name=stage_name)
    if any(not isinstance(v, str) or not v.strip() or v != v.strip()
           for v in identity.values()):
        raise ValueError("all four campaign identity fields must be nonempty and exact")
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise ValueError("campaign workspace parent must already exist as a real directory")
    target.mkdir(mode=0o700, exist_ok=False)

    repository = SQLiteStateRepository(target / "state.sqlite")
    store = FileSystemArtifactStore(target / "artifacts")
    immutable = store.put_bytes(
        canonical_bytes(_bootstrap_payload(identity)),
        media_type="application/json",
    )
    with repository.transaction() as transaction:
        transaction.append(BOOTSTRAP_COLLECTION, {
            **identity, "schema": BOOTSTRAP_SCHEMA,
            "bootstrap_artifact_ref": immutable.artifact_ref,
        })
        transaction.append(WORKFLOW_STATE_COLLECTION,
                           StagedWorkflowState(**identity).to_dict())
    workspace = CampaignWorkspace(
        target, **identity, bootstrap_artifact_ref=immutable.artifact_ref)
    observed = verify_campaign_workspace(workspace)
    if observed.verified_completed_prefix or observed.authentic_persisted_receipts:
        raise RuntimeError("fresh workspace unexpectedly claims performed science")
    return workspace


def verify_campaign_workspace(workspace: CampaignWorkspace):
    """Read-only verification of actual persisted bootstrap and stage receipts."""
    if not workspace.state_path.is_file():
        raise FileNotFoundError("original campaign SQLite state file is missing")
    repository = SQLiteStateRepository(workspace.state_path, initialize=False)
    store = FileSystemArtifactStore(workspace.artifacts_path)
    rows = [r for r in repository.all(BOOTSTRAP_COLLECTION)
            if r.get("workflow_id") == workspace.workflow_id]
    if len(rows) != 1:
        raise ValueError("campaign requires one original bootstrap record")
    row = rows[0]
    if (row.get("schema") != BOOTSTRAP_SCHEMA
            or any(row.get(k) != v for k, v in workspace.identity.items())
            or row.get("bootstrap_artifact_ref") != workspace.bootstrap_artifact_ref):
        raise ValueError("bootstrap identity/reference differs from original record")
    actual = read_verified_json(store, workspace.bootstrap_artifact_ref)
    if actual != _bootstrap_payload(workspace.identity):
        raise ValueError("bootstrap artifact has invalid original content")
    return inspect_governed_campaign(repository, store, **workspace.identity)
