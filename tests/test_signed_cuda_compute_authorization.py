"""Real Ed25519 verification tests with ephemeral UNIT-TEST-ONLY signer."""
from datetime import datetime,timedelta,timezone
from hashlib import sha256
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding,PublicFormat

from hephaestus.control.staged_autonomous import _stable_id
from hephaestus.control.staged_state import StagedApprovalRequest
from hephaestus.production.signed_cuda_grants import (
    SignedOriginalCudaGrantService,expected_gpu_operation_map,
)
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.verified import canonical_bytes,read_verified_json

SOURCE="sha256:"+"a"*64
WORKFLOW="cuda-real-genuine-45-01"
RUN="cuda-real-genuine-01"


def signed_fixture(tmp_path, *, change=None):
    private=Ed25519PrivateKey.generate() # STRICTLY unit-test-only
    public=private.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw)
    auth={
        "schema":"tier-a-original-cuda-compute-authorization.v1",
        "status":"approved","workflow_id":WORKFLOW,
        "run_id":RUN,"source_policy_ref":SOURCE,
        "device":"cuda:0","operator_id":"test-external-operator",
        "maximum_gpu_seconds":900,"external_certification":False,
        "expires_at":(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
        "operations":expected_gpu_operation_map(WORKFLOW),
    }
    if change:
        auth.update(change)
    signed={"authorization":auth,
            "signature_hex":private.sign(canonical_bytes(auth)).hex()}
    path=tmp_path/"external-signed-test-grant.json"
    path.write_bytes(canonical_bytes(signed))
    store=FileSystemArtifactStore(tmp_path/"artifacts")
    return path,public,store


def load(path,public,store,*,source=SOURCE):
    return SignedOriginalCudaGrantService.from_signed_file(
        artifacts=store,path=path,
        trusted_public_key_hex=public.hex(),
        trusted_public_key_sha256="sha256:"+sha256(public).hexdigest(),
        trusted_source_policy_ref=source,
        expected_workflow_id=WORKFLOW,expected_run_id=RUN)


def test_exact_native_gpu_op_map_includes_five_frozen_and_four_target_calls():
    ops=expected_gpu_operation_map(WORKFLOW)
    assert len(ops)==9
    assert set(ops.values())=={
        "controller","diagnosis","planner","evaluator","judge",
        "target_training","target_evaluation",
    }


def test_real_signature_binds_original_operation_source_and_gpu_ceiling(tmp_path):
    path,public,store=signed_fixture(tmp_path)
    provider=load(path,public,store)
    op=next(k for k,v in provider.authorized["operations"].items()
            if v=="target_training")
    req=StagedApprovalRequest("cuda-target:target_training:"+op,
        WORKFLOW,RUN,op,"resource_admission","target_training",SOURCE,
        ("gpu_compute","target_training","tier_a_original_research"))
    decision=provider.decision_for(req)
    assert decision.status=="approved"
    receipt=read_verified_json(store,decision.approval_ref)
    assert receipt["role"]=="target_training"
    assert receipt["maximum_gpu_seconds"]==900
    assert receipt["trusted_operator_key_sha256"]=="sha256:"+sha256(public).hexdigest()
    assert provider.decision_for(StagedApprovalRequest(
        req.request_id,req.workflow_id,req.run_id,"invented-op",
        req.phase,req.substep,req.subject_ref,req.requirements)) is None


def test_original_five_role_grant_cannot_become_training_permission(tmp_path):
    path,public,store=signed_fixture(tmp_path)
    provider=load(path,public,store)
    op=next(k for k,v in provider.authorized["operations"].items()
            if v=="controller")
    req=StagedApprovalRequest("cuda-original:"+op,WORKFLOW,RUN,op,
        "resource_admission","original_cuda_role_inference",SOURCE,
        ("gpu_compute","original_frozen_cuda_roles"))
    assert read_verified_json(store,provider.decision_for(req).approval_ref)["role"]=="controller"
    assert provider.decision_for(StagedApprovalRequest(
        "cuda-target:target_training:"+op,WORKFLOW,RUN,op,
        "resource_admission","target_training",SOURCE,
        ("gpu_compute","target_training","tier_a_original_research"))) is None


@pytest.mark.parametrize("change",[
    {"device":"cpu"},
    {"run_id":"different"},
    {"source_policy_ref":"sha256:"+"b"*64},
    {"external_certification":True},
    {"maximum_gpu_seconds":0},
    {"operator_id":"model"},
    {"operations":{"untrusted":"target_training"}},
    {"expires_at":"2001-01-01T00:00:00Z"},
])
def test_invalid_safeguard_prevents_signed_allocation(tmp_path,change):
    path,public,store=signed_fixture(tmp_path,change=change)
    with pytest.raises((PermissionError,ValueError)):
        load(path,public,store)


def test_changed_payload_and_untrusted_key_are_rejected(tmp_path):
    path,public,store=signed_fixture(tmp_path)
    payload=json.loads(path.read_bytes())
    payload["authorization"]["maximum_gpu_seconds"]=99999
    path.write_bytes(canonical_bytes(payload))
    with pytest.raises(Exception):
        load(path,public,store)
    fresh,public,store=signed_fixture(tmp_path)
    with pytest.raises(PermissionError):
        SignedOriginalCudaGrantService.from_signed_file(
            artifacts=store,path=fresh,
            trusted_public_key_hex=public.hex(),
            trusted_public_key_sha256="sha256:"+"0"*64,
            trusted_source_policy_ref=SOURCE)
