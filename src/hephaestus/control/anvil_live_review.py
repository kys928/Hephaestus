"""Off-worker deterministic evidence checks and authority issuance.

The signer must live on an independent trusted runner. Structural verification
is not semantic scientific adjudication, nor hardware-level remote attestation.
"""
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256

from hephaestus.control.anvil_attestation import AnvilAuthority
from hephaestus.control.anvil_live import ROLES, digest, publish
from hephaestus.control.anvil_live_runtime import verify_loaded_runtime
from hephaestus.llm.inference_receipts import verify_inference_receipt
from hephaestus.production.original_cuda_identity import verified_original_roles
from hephaestus.storage.verified import canonical_bytes, read_verified_json


class OffPodAuthority:
    def __init__(self, private_key):
        self._private = private_key
        public = private_key.public_key().public_bytes_raw()
        self.verifier = AnvilAuthority(public.hex(), "sha256:" + sha256(public).hexdigest())

    def sign(self, artifacts, document):
        return publish(artifacts, {"document": document,
            "signature_hex": self._private.sign(canonical_bytes(document)).hex()})


def verify_preparation(dispatch, prepared_ref, slot):
    prepared = read_verified_json(dispatch.admission.artifacts, prepared_ref, limit=32768)
    if prepared.get("schema") != "anvil.prepared-role-task.v1" or prepared.get("execution_authorized") is not False:
        raise PermissionError("expected independently prepared non-authorizing task")
    payload = read_verified_json(dispatch.admission.artifacts, prepared["dispatch_payload_ref"], limit=8192)
    plan = dispatch._plan()
    current = dispatch.admission._board.replay(plan["investigation"])
    if type(slot) is not int or not 0 <= slot < 4 or len(current.contributions) != slot or payload.get("slot") != slot:
        raise PermissionError("current hosted frontier does not permit slot")
    task = dispatch._prepare(payload)
    if canonical_bytes(asdict(task)) != canonical_bytes(prepared["task"]):
        raise PermissionError("prepared task differs from independent reconstruction")
    return task, payload


def issue_slot(authority, dispatch, *, prepared_ref, slot, runtime_result, runtime_args, expires_at):
    """The external reviewer validates the provider-bound runtime before signing."""
    if expires_at.tzinfo is None or not 0 < (expires_at - datetime.now(timezone.utc)).total_seconds() <= 3600:
        raise PermissionError("invalid bounded execution deadline")
    verify_preparation(dispatch, prepared_ref, slot)
    runtime_ref = verify_loaded_runtime(runtime_result, **runtime_args)
    publish(dispatch.admission.artifacts, runtime_result["runtime"], runtime_ref)
    return authority.sign(dispatch.admission.artifacts, {
        "schema": "anvil.execution-grant.v1", "prepared_ref": prepared_ref,
        "pod_id": runtime_args["pod_id"], "runtime_ref": runtime_ref, "device": "cuda:0",
        "seed": 271828, "max_new_tokens": 512, "expires_at": expires_at.isoformat(),
    })


def attest_retained_slot(authority, dispatch, *, slot, capsule_ref, receipt_ref, runtime_ref,
                         pod_id, retained_artifacts):
    """Review independently retrieved bytes; rejected receipts remain evidence."""
    artifacts = dispatch.admission.artifacts
    if retained_artifacts is artifacts:
        raise PermissionError("independent retained reader required")
    capsule = read_verified_json(retained_artifacts, capsule_ref)
    receipt = read_verified_json(retained_artifacts, receipt_ref)
    if capsule != read_verified_json(artifacts, capsule_ref) or receipt != read_verified_json(artifacts, receipt_ref):
        raise PermissionError("independent retained bytes differ")
    if (capsule.get("schema") != "anvil.worker-capsule.v1" or capsule.get("slot") != slot
            or capsule.get("plan_ref") != dispatch.plan_ref or capsule.get("pod_id") != pod_id
            or capsule.get("runtime_ref") != runtime_ref):
        raise PermissionError("capsule differs from independently verified worker")
    # Verify retained objects as well as the envelope, not just its digest.
    for item in capsule["objects"]:
        if digest(item["document"]) != item["ref"] or read_verified_json(artifacts, item["ref"]) != item["document"]:
            raise PermissionError("capsule object differs from review context")
    authority.verifier.verify_execution(artifacts, capsule["grant_ref"],
        prepared_ref=capsule["prepared_ref"], pod_id=pod_id, runtime_ref=runtime_ref)
    task, payload = verify_preparation(dispatch, capsule["prepared_ref"], slot)
    if capsule.get("frontier") != payload["origin"]["context_frontier"]:
        raise PermissionError("capsule frontier differs")
    stack, contract = verified_original_roles(dispatch.admission.registry_ref)
    model = stack.resolve(ROLES[slot])
    verify_inference_receipt(artifacts, receipt_ref, contract)
    expected = {"schema":"role-inference.v1", "role":task.role, "run_id":task.run_id,
        "lineage_id":task.lineage_id, "stage_name":task.stage_name, "prompt":contract.prompt(task),
        "model_id":model.model_id, "base_revision":model.revision, "adapter_sha256":model.adapter.sha256,
        "adapter_source":model.adapter.s3_key, "tokenizer_id":model.model_id,
        "tokenizer_revision":model.revision, "evidence_refs":[item.ref for item in task.evidence],
        "generator":"PinnedTransformersRoleGenerator", "seed":271828, "max_new_tokens":512}
    if receipt.get("fixture") is not None or any(receipt.get(k)!=v for k,v in expected.items()):
        raise PermissionError("raw receipt differs from pinned specialist context")
    execution = receipt.get("execution", {})
    ids = execution.get("generated_token_ids")
    runtime = read_verified_json(artifacts, runtime_ref)
    if (execution.get("device") != "cuda:0" or execution.get("gpu_name") != "NVIDIA A40"
            or execution.get("cuda_version") != runtime["loads"][0]["cuda"]
            or not isinstance(ids,list) or not 1 <= len(ids) <= 512
            or any(type(x) is not int or x < 0 for x in ids)):
        raise PermissionError("raw receipt does not describe bounded CUDA execution")
    retention_ref = publish(artifacts, {
        "schema":"anvil.receipt-retention.v1", "receipt_ref":receipt_ref,
        "capsule_ref":capsule_ref, "independent_readback":True,
    })
    if receipt.get("status") == "rejected":
        return {"status":"rejected", "retention_ref":retention_ref, "attestation_ref":None}
    if receipt.get("status") != "accepted":
        raise PermissionError("unknown receipt status")
    attestation_ref = authority.sign(artifacts, {
        "schema":"anvil.inference-attestation.v1", "prepared_ref":capsule["prepared_ref"],
        "receipt_ref":receipt_ref, "runtime_ref":runtime_ref, "plan_ref":dispatch.plan_ref,
        "pod_id":pod_id, "retention_ref":retention_ref, "status":"verified",
    })
    return {"status":"verified", "retention_ref":retention_ref, "attestation_ref":attestation_ref}


def build_review_packet(dispatch, *, slot_records, checkpoint_ref):
    """Collect inspectable scientific evidence without guessing semantic scores."""
    plan = dispatch._plan()
    snapshot = dispatch.admission._board.replay(plan["investigation"])
    checkpoint = read_verified_json(dispatch.admission.artifacts, checkpoint_ref)
    if checkpoint.get("frontier") != snapshot.frontier:
        raise PermissionError("review packet frontier differs from current hosted state")
    if len(slot_records) > 4 or any(row.get("slot") != i for i,row in enumerate(slot_records)):
        raise PermissionError("noncanonical slot evidence")
    records = []
    for i in range(4):
        slot = slot_records[i] if i < len(slot_records) else {"slot":i,"status":"missing"}
        receipt_ref = slot.get("receipt_ref")
        receipt = read_verified_json(dispatch.admission.artifacts, receipt_ref) if receipt_ref else None
        records.append({"slot":i,"role":ROLES[i],"status":slot["status"],
            "receipt_ref":receipt_ref, "receipt":receipt,
            "semantic_validity":None,"evidence_grounding":None})
    return {"schema":"anvil.scientific-review-packet.v1","plan_ref":dispatch.plan_ref,
        "frontier":snapshot.frontier,"checkpoint_ref":checkpoint_ref,"accepted_count":len(snapshot.contributions),
        "slots":records,"semantic_review_pending":True,"outcome":"unresolved",
        "execution_authorized":False}
