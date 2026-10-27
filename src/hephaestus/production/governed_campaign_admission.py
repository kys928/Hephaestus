"""Read-only verified-prefix admission for the *existing* governed 45-operation run.

This is NOT a workflow generator. It refuses missing historical state, wrong
identity, gaps, and unauthenticated receipts. It identifies the *exact*
persisted approval request so an operator/provider can make a scoped decision.

Only the production composer may supply genuine real-role services. The
caller can resume the existing orchestrator without changing phase order or
approval policy; this module never manufactures an approval or phase result.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from hephaestus.control.spine import SPINE_ORDER
from hephaestus.control.staged_state import (
    GOVERNED_AUTONOMOUS_MODE, PHASE_SUBSTEPS, REQUIRED_RECORD_KINDS,
    WORKFLOW_RECORD_COLLECTION, WORKFLOW_STATE_COLLECTION,
    StagedAutonomousDependencies, StagedWorkflowState,
)
from hephaestus.control.staged_autonomous import build_staged_autonomous_orchestrator
from hephaestus.storage.verified import read_verified_json


_ORDER = tuple(f"{phase.value}.{substep}" for phase in SPINE_ORDER
               for substep in PHASE_SUBSTEPS[phase])
if len(_ORDER) != 45 or len(set(_ORDER)) != 45:
    raise RuntimeError("original governed 45-step contract has drifted")


@dataclass(frozen=True)
class GovernedCampaignAdmission:
    workflow_id: str
    run_id: str
    lineage_id: str
    stage_name: str
    verified_completed_prefix: int
    next_operation: str | None
    pending_approval_request: dict[str, object] | None
    original_workflow_status: str
    resumable: bool
    blocking_issues: tuple[str, ...]
    authentic_persisted_receipts: int
    completed_campaign: bool
    certified_role_execution_proven: bool = False
    dataset_training_approval_conferred: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "hephaestus-governed-45-prefix-admission.v1",
            "workflow_id": self.workflow_id,
            "run_id": self.run_id,
            "lineage_id": self.lineage_id,
            "stage_name": self.stage_name,
            "verified_completed_prefix": self.verified_completed_prefix,
            "next_operation": self.next_operation,
            "pending_approval_request": self.pending_approval_request,
            "original_workflow_status": self.original_workflow_status,
            "resumable": self.resumable,
            "blocking_issues": list(self.blocking_issues),
            "authentic_persisted_receipts": self.authentic_persisted_receipts,
            "completed_campaign": self.completed_campaign,
            "certified_role_execution_proven": False,
            "dataset_training_approval_conferred": False,
            "inspection_did_not_advance_campaign": True,
        }


def inspect_governed_campaign(
    repository, artifacts, *, workflow_id: str, run_id: str,
    lineage_id: str, stage_name: str,
) -> GovernedCampaignAdmission:
    """Prove an unbroken prefix from original persisted bytes, not documents.

    A green inspection authenticates *only* the staged control receipts. It
    cannot independently certify workers' weights, approvals, or data rights.
    """
    persisted = repository.get_latest(
        WORKFLOW_STATE_COLLECTION, "workflow_id", workflow_id)
    if persisted is None:
        raise FileNotFoundError("no persisted governed campaign with this workflow_id")
    state = StagedWorkflowState.from_dict(persisted)
    if (state.workflow_id, state.run_id, state.lineage_id, state.stage_name) != (
        workflow_id, run_id, lineage_id, stage_name
    ):
        raise PermissionError("governed campaign identity disagrees with stored workflow")
    if state.mode != GOVERNED_AUTONOMOUS_MODE or state.phase_order != [
        phase.value for phase in SPINE_ORDER
    ]:
        raise ValueError("stored mode or original ordered eight-phase spine changed")
    if not set(state.steps).issubset(_ORDER):
        raise ValueError("persisted campaign has an unknown operation")
    if state.completion_marker and (state.status != "completed" or
                                   len(state.steps) != 45):
        raise ValueError("stored completion marker is inconsistent")

    scoped_rows = [r for r in repository.all(WORKFLOW_RECORD_COLLECTION)
                   if r.get("workflow_id") == workflow_id]
    completed = 0
    verified_receipts = 0
    first_incomplete = None
    for ordinal, key in enumerate(_ORDER):
        step = state.steps.get(key)
        if step is None:
            first_incomplete = ordinal
            break
        phase, substep = key.split(".", 1)
        if (step.phase, step.substep) != (phase, substep):
            raise ValueError("stored operation key differs from state identity")
        if not step.operation_id or step.attempt < 1:
            raise ValueError("persisted operation lacks real attempt identity")
        if step.completion_marker != (step.status == "completed"):
            raise ValueError("inconsistent operation completion marker")
        expected_receipt = {
            "status": step.status, "attempt": step.attempt,
            "resumable": step.resumable,
            "completion_marker": step.completion_marker,
            "input_refs": step.input_refs, "output_refs": step.output_refs,
            "blocking_issues": step.blocking_issues,
        }
        matched = 0
        for record in scoped_rows:
            if (record.get("phase"), record.get("substep"),
                    record.get("operation_id"), record.get("kind")) != (
                phase, substep, step.operation_id, "substep_result"
            ):
                continue
            if any(record.get(k) != v for k, v in (
                ("workflow_id", workflow_id), ("run_id", run_id),
                ("lineage_id", lineage_id), ("stage_name", stage_name)
            )):
                raise ValueError("operation receipt has foreign workflow identity")
            ref = record.get("artifact_ref")
            if not isinstance(ref, str) or not ref.startswith("sha256:"):
                raise ValueError("operation receipt lacks content-addressed original bytes")
            actual = read_verified_json(artifacts, ref)
            if actual != {k: v for k, v in record.items() if k != "artifact_ref"}:
                raise ValueError("persisted operation row differs from original bytes")
            if record.get("payload") == expected_receipt:
                matched += 1
        if matched != 1:
            raise ValueError(f"operation receipt not uniquely verified: {key}")
        # The control receipt alone is necessary but not sufficient: a stage
        # that claims scientific work must have its *actual* recorded science
        # output on the SAME operation identity, with immutable source bytes.
        critical_kind = REQUIRED_RECORD_KINDS.get(substep)
        if step.completion_marker and critical_kind:
            if critical_kind not in step.output.get("record_kinds", []):
                raise ValueError(f"completed scientific stage lacks declared original record: {key}")
            records = [
                r for r in scoped_rows
                if (r.get("phase"), r.get("substep"), r.get("operation_id"),
                    r.get("kind")) == (phase, substep, step.operation_id, critical_kind)
            ]
            if len(records) != 1:
                raise ValueError(f"scientific stage has no unique original source record: {key}")
            critical = records[0]
            source_ref = critical.get("artifact_ref")
            if not isinstance(source_ref, str) or not source_ref.startswith("sha256:"):
                raise ValueError(f"scientific stage has no hashed original source record: {key}")
            if read_verified_json(artifacts, source_ref) != {
                k: v for k, v in critical.items() if k != "artifact_ref"
            }:
                raise ValueError(f"scientific source record bytes differ: {key}")
        verified_receipts += 1
        if not step.completion_marker:
            first_incomplete = ordinal
            break
        completed += 1

    # A completed prefix is *consecutive* and must not be followed by
    # already-completed steps, including forged ones in an otherwise blocked run.
    for key in _ORDER[completed+1:]:
        if key in state.steps and state.steps[key].completion_marker:
            raise ValueError("out-of-order completed stage violates genuine spine")
    if first_incomplete is None and completed != len(_ORDER):
        first_incomplete = completed
    if state.completion_marker and completed != len(_ORDER):
        raise ValueError("full completion claim has incomplete actual receipts")

    next_step = _ORDER[completed] if completed < len(_ORDER) else None
    pending = state.steps.get(next_step) if next_step else None
    request = dict(pending.approval_request) if pending and pending.approval_request else None
    issues = tuple(pending.blocking_issues if pending and pending.blocking_issues else state.blocking_issues)
    if (next_step and next_step.endswith(".acquisition_approval_gate")
            and pending and pending.status == "approval_pending"):
        if not request or request.get("substep") != "acquisition_approval_gate":
            raise ValueError("pending acquisition approval lacks genuine persisted request")
        if (request.get("workflow_id") != workflow_id
                or request.get("run_id") != run_id
                or request.get("operation_id") != pending.operation_id):
            raise ValueError("persisted approval request does not belong to staged gate")
    return GovernedCampaignAdmission(
        workflow_id=workflow_id, run_id=run_id, lineage_id=lineage_id,
        stage_name=stage_name,
        verified_completed_prefix=completed, next_operation=next_step,
        pending_approval_request=request,
        original_workflow_status=state.status,
        resumable=state.resumable,
        blocking_issues=issues,
        authentic_persisted_receipts=verified_receipts,
        completed_campaign=bool(state.completion_marker and completed == 45),
    )


def resume_real_governed_campaign(
    dependencies: StagedAutonomousDependencies, *, workflow_id: str,
    run_id: str, lineage_id: str, stage_name: str,
):
    """Resume the exact original persisted orchestrator, NEVER synthesize 11.

    Requires all real service bindings and certified CUDA role runtime. The
    existing native approval gate makes the source decision, not this adapter.
    """
    if dependencies.artifact_store is None or dependencies.approval_service is None:
        raise PermissionError("content-addressed custody and scoped approval provider required")
    from hephaestus.production.staged_composition import staged_binding_inventory
    from hephaestus.production.model_roles import ModelBackedStage
    from hephaestus.llm.certified_runtime import (
        CertifiedRoleReasoner, PinnedTransformersRoleGenerator,
    )
    # A CPU advisory scorer is NOT five distinct frozen role masters. Require
    # exactly the same certified CUDA topology as production composition;
    # staged_binding_inventory alone authenticates service signatures only.
    role_fields = (
        ("controller", "judge_entry"), ("diagnosis", "diagnosis"),
        ("planner", "planner"), ("evaluator", "evaluator"),
        ("judge", "judge_exit"),
    )
    roles = [getattr(dependencies.services, field) for _, field in role_fields]
    if not all(
        type(service) is ModelBackedStage and service.role == role
        for (role, _), service in zip(role_fields, roles, strict=True)
    ):
        raise PermissionError("five certified model-backed scientific role services required")
    shared = roles[0].reasoner
    if (type(shared) is not CertifiedRoleReasoner
            or any(service.reasoner is not shared for service in roles)
            or type(shared.generator) is not PinnedTransformersRoleGenerator
            or shared.generator.device != "cuda:0"
            or not shared.stack.production_certified
            or not shared.stack.runtime_registry_ready):
        raise PermissionError("frozen genuine five-role CUDA runtime is not admitted")
    # Force native check of all 45 *real* production adapters before state IO.
    staged_binding_inventory(dependencies)
    admission = inspect_governed_campaign(
        dependencies.state_repository, dependencies.artifact_store,
        workflow_id=workflow_id, run_id=run_id,
        lineage_id=lineage_id, stage_name=stage_name)
    if admission.completed_campaign:
        return admission
    if not admission.resumable:
        raise PermissionError("original campaign is not resumable")
    if admission.verified_completed_prefix == 0:
        raise PermissionError("new campaigns require separate launch authorization")
    worker = build_staged_autonomous_orchestrator(
        workflow_id=workflow_id, run_id=run_id, lineage_id=lineage_id,
        stage_name=stage_name, dependencies=dependencies)
    worker.resume()  # The existing native gate checks any current grant.
    return inspect_governed_campaign(
        dependencies.state_repository, dependencies.artifact_store,
        workflow_id=workflow_id, run_id=run_id,
        lineage_id=lineage_id, stage_name=stage_name)
