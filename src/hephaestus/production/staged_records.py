"""Shared source-of-truth reader for immutable staged workflow records."""
from __future__ import annotations

from hephaestus.control.staged_state import StagedOperationRequest, WORKFLOW_RECORD_COLLECTION
from hephaestus.storage.base import StateRepository, ArtifactStore
from hephaestus.storage.verified import read_verified_json


def verified_workflow_payload(repository: StateRepository, artifacts: ArtifactStore,
                              request: StagedOperationRequest, *,
                              substep: str, kind: str) -> dict[str, object]:
    rows = repository.all(WORKFLOW_RECORD_COLLECTION)
    matching = [r for r in rows if (
        r.get("workflow_id") == request.workflow_id
        and r.get("run_id") == request.run_id
        and r.get("lineage_id") == request.lineage_id
        and r.get("stage_name") == request.stage_name
        and r.get("substep") == substep
        and r.get("kind") == kind
    )]
    if len(matching) != 1:
        raise ValueError(f"expected one immutable {substep}/{kind} record")
    row = matching[0]
    ref = row.get("artifact_ref")
    if not isinstance(ref, str):
        raise ValueError("scientific workflow record missing immutable artifact reference")
    saved = read_verified_json(artifacts, ref)
    if saved != {k: v for k, v in row.items() if k != "artifact_ref"}:
        raise ValueError("persisted workflow row and immutable bytes diverge")
    payload = saved.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("workflow record payload is not a JSON object")
    return payload
