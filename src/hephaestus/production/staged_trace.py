"""Content-addressed trace of every completed staged operation.

A verified receipt chain proves that the control spine persisted each result.
It does NOT prove that a worker used real weights or that a model improved.
Scientific claims require checkpoint/inference and independent holdout proofs.
"""
from __future__ import annotations

from hephaestus.control.spine import SPINE_ORDER
from hephaestus.control.staged_state import (
    PHASE_SUBSTEPS, REQUIRED_RECORD_KINDS, WORKFLOW_RECORD_COLLECTION,
    StagedWorkflowState,
)
from hephaestus.storage.verified import canonical_bytes, read_verified_json


def verify_completed_staged_trace(workflow: StagedWorkflowState, repository, artifacts) -> str:
    """Reverify exact persisted operation-result bytes for a complete cycle."""
    expected = [
        (phase.value, substep)
        for phase in SPINE_ORDER
        for substep in PHASE_SUBSTEPS[phase]
    ]
    expected_keys = {f"{phase}.{step}" for phase, step in expected}
    if (not workflow.completion_marker or workflow.status != "completed"
            or len(expected) != 45 or set(workflow.steps) != expected_keys):
        raise ValueError("45-operation staged workflow is not complete")
    rows = [
        row for row in repository.all(WORKFLOW_RECORD_COLLECTION)
        if row.get("workflow_id") == workflow.workflow_id
    ]
    seen_ids: set[str] = set()
    evidence = []
    for phase, substep in expected:
        step = workflow.steps[f"{phase}.{substep}"]
        if (step.phase != phase or step.substep != substep
                or not step.operation_id or step.operation_id in seen_ids
                or step.status != "completed" or not step.completion_marker):
            raise ValueError("staged operation identity or completion invalid")
        seen_ids.add(step.operation_id)
        matching = [
            row for row in rows
            if (row.get("operation_id") == step.operation_id
                and row.get("phase") == phase
                and row.get("substep") == substep)
        ]
        receipts = []
        for row in matching:
            ref = row.get("artifact_ref")
            if not isinstance(ref, str) or not ref.startswith("sha256:"):
                raise ValueError("staged operation lacks hash-backed receipt")
            recovered = read_verified_json(artifacts, ref)
            if recovered != {key: value for key, value in row.items() if key != "artifact_ref"}:
                raise ValueError("staged receipt diverges from immutable bytes")
            if any(row.get(k) != v for k, v in (
                    ("workflow_id", workflow.workflow_id),
                    ("run_id", workflow.run_id),
                    ("lineage_id", workflow.lineage_id),
                    ("stage_name", workflow.stage_name))):
                raise ValueError("operation receipt belongs to a different scientific run")
            receipts.append((row["kind"], ref, row["payload"]))
        completed = [
            (ref, payload) for kind, ref, payload in receipts
            if kind == "substep_result" and payload.get("status") == "completed"
            and payload.get("attempt") == step.attempt
            and payload.get("completion_marker") is True
        ]
        if len(completed) != 1:
            raise ValueError("completed staged operation requires exactly one final receipt")
        if (completed[0][1].get("input_refs") != step.input_refs
                or completed[0][1].get("output_refs") != step.output_refs):
            raise ValueError("operation receipt and workflow state disagree")
        required_kind = REQUIRED_RECORD_KINDS.get(substep)
        if required_kind and not any(kind == required_kind for kind, _, _ in receipts):
            raise ValueError(f"missing required decision-critical receipt: {substep}")
        evidence.append({
            "phase": phase, "substep": substep,
            "operation_id": step.operation_id, "attempt": step.attempt,
            "completion_ref": completed[0][0],
            "required_record_kind": required_kind,
            "record_refs": sorted({ref for _, ref, _ in receipts}),
        })
    document = {
        "schema": "staged-45-operation-trace.v1",
        "workflow_id": workflow.workflow_id,
        "run_id": workflow.run_id,
        "lineage_id": workflow.lineage_id,
        "stage_name": workflow.stage_name,
        "operations": evidence,
        "verified_operation_count": len(evidence),
        "scope": "control_record_integrity_only_not_real_model_proof",
        "scientific_certification": "not_certified",
    }
    return artifacts.put_bytes(canonical_bytes(document), media_type="application/json").artifact_ref
