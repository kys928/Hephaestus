"""Run a prepared original specialist under an independent per-slot grant."""

from dataclasses import asdict, replace
from hashlib import sha256
import os

from hephaestus.control.anvil_attestation import AnvilAuthority
from hephaestus.control.anvil_dispatch import AnvilDispatch
from hephaestus.control.deterministic_role_boundary import (
    DeterministicRoleContext, EvidenceRecord, EvidenceRegistry,
)
from hephaestus.llm.certified_runtime import CertifiedRoleReasoner, PinnedTransformersRoleGenerator
from hephaestus.production.original_cuda_identity import verified_original_roles
from hephaestus.schemas.anvil import ContributionOrigin, require_digest
from hephaestus.state.execution_ownership import execution_owner
from hephaestus.storage.verified import canonical_bytes, read_verified_json


def _require_open(repository, plan_ref: str) -> None:
    if any(row["plan_ref"] == plan_ref for row in repository.all("anvil.terminal")):
        raise PermissionError("Anvil investigation already has a terminal review")


def execute_prepared(
    dispatch: AnvilDispatch, *, prepared_ref: str, grant_ref: str, runtime_ref: str,
    authority: AnvilAuthority, reasoner: CertifiedRoleReasoner, reservations,
):
    """Execute on an existing worker. Return raw receipt identity for independent review."""
    artifacts = dispatch.admission.artifacts
    authority.verify_execution(artifacts, grant_ref, prepared_ref=prepared_ref,
                               pod_id=os.environ.get("RUNPOD_POD_ID", ""), runtime_ref=runtime_ref)
    prepared = read_verified_json(artifacts, prepared_ref, limit=32768)
    if prepared.get("schema") != "anvil.prepared-role-task.v1":
        raise ValueError("incorrect prepared task schema")
    payload = read_verified_json(artifacts, prepared["dispatch_payload_ref"], limit=8192)
    task = dispatch._prepare(payload)
    if (canonical_bytes(prepared["task"]) != canonical_bytes(asdict(task))
            or type(reasoner) is not CertifiedRoleReasoner
            or type(reasoner.generator) is not PinnedTransformersRoleGenerator
            or reasoner.generator.device != "cuda:0"
            or reasoner.generator.allow_remote_pinned_download
            or reasoner.artifact_store is not artifacts):
        raise PermissionError("execution differs from the prepared offline CUDA task")
    stack, role_contract = verified_original_roles(dispatch.admission.registry_ref)
    if (reasoner.stack.resolve(task.role) != stack.resolve(task.role)
            or reasoner.contract.payload != role_contract.payload):
        raise PermissionError("reasoner does not match the certified role registry and contract")
    reservation = {"slot": payload["slot"], "plan_ref": dispatch.plan_ref,
                   "prepared_ref": prepared_ref, "grant_ref": grant_ref}
    key = "sha256:" + sha256(canonical_bytes([dispatch.plan_ref, payload["slot"]])).hexdigest()
    with execution_owner(reservations, "anvil-investigation", dispatch.plan_ref):
        _require_open(reservations, dispatch.plan_ref)
        if any(row["key"] == key for row in reservations.all("anvil.execution-slots")):
            raise PermissionError("Anvil execution slot already consumed")
        reservations.append("anvil.execution-slots", dict(reservation, key=key), operation_id=key)
    registry = EvidenceRegistry({
        item.ref: EvidenceRecord(item.ref, "anvil_material", task.run_id, task.lineage_id)
        for item in task.evidence
    })
    context = DeterministicRoleContext(task.run_id, task.lineage_id, task.stage_name)
    return reasoner.propose(task, registry, context=context, seed=271828, max_new_tokens=512)


def admit_attested(
    dispatch: AnvilDispatch, *, prepared_ref: str, receipt_ref: str,
    runtime_ref: str, attestation_ref: str, authority: AnvilAuthority, repository,
):
    """Bind independently reviewed runtime evidence to the exact prepared claim."""
    require_digest(runtime_ref)
    require_digest(receipt_ref)
    artifacts = dispatch.admission.artifacts
    authority.verify(artifacts, attestation_ref, schema="anvil.inference-attestation.v1", expected={
        "prepared_ref": prepared_ref, "receipt_ref": receipt_ref, "runtime_ref": runtime_ref,
        "plan_ref": dispatch.plan_ref, "status": "verified",
    })
    prepared = read_verified_json(artifacts, prepared_ref, limit=32768)
    if prepared.get("schema") != "anvil.prepared-role-task.v1":
        raise ValueError("incorrect prepared task schema")
    payload = read_verified_json(artifacts, prepared["dispatch_payload_ref"], limit=8192)
    dispatch._prepare(payload)
    origin = replace(ContributionOrigin(**payload["origin"]), inference_ref=receipt_ref)
    with execution_owner(repository, "anvil-investigation", dispatch.plan_ref):
        _require_open(repository, dispatch.plan_ref)
        return dispatch.admission.accept(origin, payload["sequence"], tuple(payload["material_refs"]))


def retain_terminal(
    dispatch: AnvilDispatch, *, terminal_ref: str, frontier: str,
    authority: AnvilAuthority, repository,
):
    """Retain a reviewer-signed outcome without converting advice into execution authority."""
    require_digest(frontier)
    artifacts = dispatch.admission.artifacts
    plan = dispatch._plan()
    scope = [plan["investigation"], plan["run_id"], plan["lineage_id"]]
    decision = authority.verify(artifacts, terminal_ref, schema="anvil.terminal.v1", expected={
        "plan_ref": dispatch.plan_ref, "frontier": frontier, "execution_authorized": False,
    })
    if decision.get("outcome") not in {"supported_agreement", "justified_disagreement", "unresolved"}:
        raise ValueError("Anvil terminal outcome is outside the preregistered rubric")
    review_ref = decision.get("review_ref")
    require_digest(review_ref)
    read_verified_json(artifacts, review_ref, limit=32768)
    identity = "sha256:" + sha256(canonical_bytes(["anvil.terminal.v1", scope])).hexdigest()
    with execution_owner(repository, "anvil-investigation", dispatch.plan_ref):
        snapshot = dispatch.admission._board.replay(plan["investigation"], expected_frontier=frontier)
        if any([row.origin.run_id, row.origin.lineage_id] != scope[1:] for row in snapshot.contributions):
            raise ValueError("terminal frontier mixes investigation scopes")
        return repository.append("anvil.terminal", {
            "plan_ref": dispatch.plan_ref, "terminal_ref": terminal_ref, "frontier": frontier,
            "outcome": decision["outcome"], "review_ref": review_ref,
        }, operation_id=identity)
