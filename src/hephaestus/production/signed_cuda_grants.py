"""Independent, operator-signed, strictly bounded original CUDA resource grants.

An authenticated *external* Ed25519 authorization manifest is required before
any GPU role, target training, or target evaluation. The training model,
RunPod worker and this program cannot sign or approve their own execution.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import re

from hephaestus.control.staged_autonomous import _stable_id
from hephaestus.control.staged_state import (
    PHASE_SUBSTEPS, StagedApprovalDecision, StagedApprovalRequest,
)
from hephaestus.control.spine import SPINE_ORDER
from hephaestus.storage.verified import canonical_bytes

_SHA=re.compile(r"sha256:[0-9a-f]{64}\Z")
_ROLE_KEYS={
    "judge_entry.judge_entry_decision":"controller",
    "judge_entry.evidence_based_diagnosis":"diagnosis",
    "planner.intervention_ranking":"planner",
    "evaluator.semantic_comparison":"evaluator",
    "judge_exit.governed_verdict":"judge",
}
_EVALUATION_KEYS={
    "judge_entry.evidence_collection",
    "evaluator.candidate_generation",
    "evaluator.repeatability_variance_evidence",
}


def expected_gpu_operation_map(workflow_id):
    """Exact original 45-op native graph; no caller-defined extension."""
    if not isinstance(workflow_id,str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}",workflow_id):
        raise PermissionError("GPU operation map requires exact original workflow ID")
    ops={}
    for phase in SPINE_ORDER:
        for substep in PHASE_SUBSTEPS[phase]:
            name=f"{phase.value}.{substep}"
            op="op-"+_stable_id(workflow_id,phase.value,substep)
            if name in _ROLE_KEYS:
                ops[op]=_ROLE_KEYS[name]
            elif name=="training_engineer.lifecycle_launch":
                ops[op]="target_training"
            elif name in _EVALUATION_KEYS:
                ops[op]="target_evaluation"
    if len(ops)!=9 or set(ops.values())!={"controller","diagnosis",
            "planner","evaluator","judge","target_training","target_evaluation"}:
        raise RuntimeError("original native GPU role/target operation map drifted")
    return ops


@dataclass(slots=True)
class SignedOriginalCudaGrantService:
    """Production-only public-key verifier; no private key or self-sign path."""
    artifacts: object
    authorized: dict
    signature_hex: str
    key_hex: str
    trusted_key_sha256: str

    @classmethod
    def from_signed_file(cls, *, artifacts, path: Path, trusted_public_key_hex: str,
                         trusted_public_key_sha256: str, trusted_source_policy_ref: str,
                         expected_workflow_id: str, expected_run_id: str):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        raw=bytes.fromhex(trusted_public_key_hex)
        if len(raw)!=32 or "sha256:"+sha256(raw).hexdigest()!=trusted_public_key_sha256:
            raise PermissionError("independently pinned trusted CUDA owner public key mismatch")
        data=json.loads(Path(path).read_bytes())
        if set(data)!={"authorization","signature_hex"}:
            raise PermissionError("CUDA grant requires exact signed authorization envelope")
        document,signature=data["authorization"],bytes.fromhex(data["signature_hex"])
        Ed25519PublicKey.from_public_bytes(raw).verify(signature,canonical_bytes(document))
        ops=expected_gpu_operation_map(document.get("workflow_id"))
        until=datetime.fromisoformat(document["expires_at"].replace("Z","+00:00"))
        now=datetime.now(timezone.utc)
        if (document.get("schema")!="tier-a-original-cuda-compute-authorization.v1"
                or document.get("workflow_id")!=expected_workflow_id
                or document.get("run_id")!=expected_run_id
                or document.get("source_policy_ref")!=trusted_source_policy_ref
                or not _SHA.fullmatch(trusted_source_policy_ref)
                or document.get("device")!="cuda:0"
                or document.get("status")!="approved"
                or document.get("external_certification") is not False
                or document.get("operator_id") in ("",None,"self","model")
                or type(document.get("maximum_gpu_seconds")) is not int
                or not 0<document["maximum_gpu_seconds"]<=86400
                or until.tzinfo is None or until<=now
                or not isinstance(document.get("operations"),dict)
                or document["operations"]!=ops):
            raise PermissionError("externally signed original CUDA compute authorization not valid")
        return cls(artifacts,document,data["signature_hex"],
                   trusted_public_key_hex,trusted_public_key_sha256)

    def decision_for(self, request: StagedApprovalRequest):
        a=self.authorized
        if (request.workflow_id!=a["workflow_id"] or request.run_id!=a["run_id"]
                or request.subject_ref!=a["source_policy_ref"]):
            return None
        try:
            expires=datetime.fromisoformat(a["expires_at"].replace("Z","+00:00"))
        except (ValueError,TypeError):
            return None
        if expires<=datetime.now(timezone.utc):
            return None
        purpose=a["operations"].get(request.operation_id)
        if purpose is None:
            return None
        if purpose in {"target_training","target_evaluation"}:
            required={"gpu_compute",purpose,"tier_a_original_research"}
            permitted=(request.phase=="resource_admission" and request.substep==purpose
                       and request.request_id=="cuda-target:"+purpose+":"+request.operation_id)
        else:
            required={"gpu_compute","original_frozen_cuda_roles"}
            permitted=(request.phase=="resource_admission" and
                request.substep=="original_cuda_role_inference" and
                request.request_id=="cuda-original:"+request.operation_id)
        if not permitted or set(request.requirements)!=required:
            return None
        receipt={
            "schema":"original-cuda-resource-grant.v1",
            "workflow_id":request.workflow_id,
            "run_id":request.run_id,
            "operation_id":request.operation_id,
            "source_policy_ref":request.subject_ref,
            "role":purpose,"device":"cuda:0",
            "status":"approved","authorized_by":a["operator_id"],
            "maximum_gpu_seconds":a["maximum_gpu_seconds"],
            "external_scientific_certification":False,
            "external_authorization_sha256":"sha256:"+sha256(canonical_bytes(a)).hexdigest(),
            "external_ed25519_signature_hex":self.signature_hex,
            "trusted_operator_key_sha256":self.trusted_key_sha256,
        }
        ref=self.artifacts.put_bytes(canonical_bytes(receipt),media_type="application/json").artifact_ref
        return StagedApprovalDecision(
            request.request_id,request.operation_id,request.subject_ref,"approved",ref,
            tuple(request.requirements))
