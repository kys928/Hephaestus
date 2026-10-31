"""Operator-signed CUDA request: offline private key never reaches the GPU worker."""
import importlib.util
import os
from pathlib import Path
from datetime import datetime,timedelta,timezone
from hashlib import sha256
import json

import pytest
pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding,PrivateFormat,NoEncryption
from hephaestus.production.signed_cuda_grants import (
    SignedOriginalCudaGrantService,expected_gpu_operation_map,
)
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.verified import canonical_bytes

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location(
    "offline_original_cuda_signer",ROOT/"scripts/operator_sign_original_cuda_request.py")
offline=importlib.util.module_from_spec(spec)
spec.loader.exec_module(offline)

WF="cuda-original-operator-request-test"
RUN="cuda-run-operator-test"
SOURCE="sha256:"+"e"*64


def fixtures(tmp_path):
    key=Ed25519PrivateKey.generate() # transient unit-test key only; never production authorization
    seed=key.private_bytes(Encoding.Raw,PrivateFormat.Raw,NoEncryption())
    secret=tmp_path/"user-owned-private-key.txt"
    secret.write_text(seed.hex())
    secret.chmod(0o600)
    payload={
        "schema":"tier-a-original-cuda-compute-authorization.v1",
        "status":"requested","workflow_id":WF,"run_id":RUN,
        "source_policy_ref":SOURCE,
        "reviewed_trainable_ref":"sha256:"+"c"*64,
        "device":"cuda:0","maximum_gpu_seconds":900,
        "operations":expected_gpu_operation_map(WF),
        "operator_id":"REQUIRES_EXTERNAL_OPERATOR_ID",
        "expires_at":"REQUIRES_UTC_EXPIRY_TIME",
        "external_certification":False,
    }
    req=tmp_path/"original-cuda-unsigned.json"
    req.write_bytes(canonical_bytes(payload))
    return secret,req


def test_offline_operator_signing_yields_real_verifiable_scoped_grant(tmp_path):
    secret,req=fixtures(tmp_path)
    until=(datetime.now(timezone.utc)+timedelta(minutes=20)).isoformat()
    receipt=offline.sign_request(req,private_key_file=secret,
        operator_id="test-scientific-budget-owner",expires_at=until,
        output=tmp_path/"original-cuda-approved.json")
    assert receipt["independent_scientific_certification"] is False
    assert receipt["gpu_authorized"] is True
    signed_file=tmp_path/"original-cuda-approved.json"
    assert signed_file.stat().st_mode & 0o077==0
    store=FileSystemArtifactStore(tmp_path/"artifacts")
    provider=SignedOriginalCudaGrantService.from_signed_file(
        artifacts=store,path=signed_file,
        trusted_public_key_hex=receipt["trusted_public_key_hex"],
        trusted_public_key_sha256=receipt["trusted_public_key_sha256"],
        trusted_source_policy_ref=SOURCE,
        expected_workflow_id=WF,expected_run_id=RUN)
    assert provider.authorized["operator_id"]=="test-scientific-budget-owner"
    assert provider.authorized["reviewed_trainable_ref"]=="sha256:"+"c"*64


def test_operator_rejects_unsafe_private_key_and_self_grants(tmp_path):
    secret,req=fixtures(tmp_path)
    until=(datetime.now(timezone.utc)+timedelta(minutes=20)).isoformat()
    secret.chmod(0o644)
    with pytest.raises(PermissionError):
        offline.sign_request(req,private_key_file=secret,
            operator_id="test-budget-owner",expires_at=until,
            output=tmp_path/"no.json")
    secret.chmod(0o600)
    with pytest.raises(PermissionError):
        offline.sign_request(req,private_key_file=secret,
            operator_id="model",expires_at=until,
            output=tmp_path/"no.json")
    forged=json.loads(req.read_bytes());forged["status"]="approved"
    req.write_bytes(canonical_bytes(forged))
    with pytest.raises(PermissionError):
        offline.sign_request(req,private_key_file=secret,
            operator_id="external-person",expires_at=until,
            output=tmp_path/"no.json")
