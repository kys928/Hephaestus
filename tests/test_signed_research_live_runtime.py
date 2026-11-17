"""Independent-signer CUDA admission for genuine frozen Diagnostician research.

Key generation here is a UNIT TEST ONLY; deployment must receive the public
key through a different administrator, never the model/worker itself.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
import json

import pytest

from hephaestus.research.live_runtime import (
    SignedResearchComputeGrant, compose_frozen_diagnosis_research,
)
from hephaestus.production.original_cuda_identity import registry_digest
from hephaestus.storage.verified import canonical_bytes
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.sqlite import SQLiteStateRepository


def _signed_grant(tmp_path, **overrides):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    key=Ed25519PrivateKey.generate()
    public=key.public_key().public_bytes(serialization.Encoding.Raw,
                                            serialization.PublicFormat.Raw)
    expected_catalog="sha256:"+"a"*64
    doc={
        "schema":"hephaestus-research-compute-grant.v1",
        "status":"approved","operator_id":"independent-operator",
        "run_id":"research-run","lineage_id":"lineage",
        "catalog_sha256":expected_catalog,
        "role_registry_sha256":registry_digest(),
        "role":"diagnosis","provider":"runpod","region":"EU-CZ-1",
        "gpu_sku":"RTX-A6000","device":"cuda:0",
        "maximum_inference_calls":6,"maximum_new_tokens":384,
        "maximum_gpu_seconds":1200,
        "maximum_cost_usd_cents":120,
        "maximum_hourly_price_usd_cents":350,
        "expires_at":(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
        "approval_scope":"single_evaluation_only_study",
        "training_authorized":False,"code_edit_authorized":False,
        "model_promotion_authorized":False,
    }
    doc.update(overrides)
    sig=key.sign(canonical_bytes(doc))
    path=tmp_path/"external-signature.json"
    path.write_bytes(canonical_bytes({"authorization":doc,"signature_hex":sig.hex()}))
    return path,public.hex(),"sha256:"+sha256(public).hexdigest(),expected_catalog


def _load(tmp_path,**overrides):
    path,key,digest,catalog=_signed_grant(tmp_path,**overrides)
    return SignedResearchComputeGrant.load(
        path,public_key_hex=key,independently_pinned_key_sha256=digest,
        expected_run_id="research-run",expected_lineage_id="lineage",
        expected_catalog_sha256=catalog,expected_role_registry_sha256=registry_digest())


def test_valid_independently_signed_cuda_study_is_narrow(tmp_path):
    grant=_load(tmp_path)
    grant.check(run_id="research-run",lineage_id="lineage",calls=6,
                catalog_sha256="sha256:"+"a"*64,registry_sha256=registry_digest())
    for args in [
        {"run_id":"other"},
        {"calls":7},
        {"lineage_id":"other"},
        {"catalog_sha256":"sha256:"+"b"*64},
    ]:
        request={"run_id":"research-run","lineage_id":"lineage","calls":6,
                 "catalog_sha256":"sha256:"+"a"*64,"registry_sha256":registry_digest()}
        request.update(args)
        with pytest.raises(PermissionError):
            grant.check(**request)


@pytest.mark.parametrize("forbidden",[
    {"training_authorized":True},{"code_edit_authorized":True},
    {"model_promotion_authorized":True},{"maximum_inference_calls":120},
    {"maximum_new_tokens":4096},{"maximum_gpu_seconds":86400},
    {"maximum_cost_usd_cents":0},{"maximum_cost_usd_cents":1},
    {"maximum_hourly_price_usd_cents":11000},{"maximum_hourly_price_usd_cents":0},
    {"maximum_gpu_seconds":3600},
    {"device":"cpu"},{"role":"evaluator"},{"provider":"local"},
    {"operator_id":"model"},
    {"expires_at":(datetime.now(timezone.utc)-timedelta(days=1)).isoformat()},
])
def test_signed_grant_refuses_other_powers_and_oversized_spending(tmp_path,forbidden):
    with pytest.raises(PermissionError):
        _load(tmp_path,**forbidden)


def test_fake_trust_anchor_cannot_approve_even_valid_signed_manifest(tmp_path):
    path,key,digest,catalog=_signed_grant(tmp_path)
    with pytest.raises(PermissionError):
        SignedResearchComputeGrant.load(path,public_key_hex=key,
            independently_pinned_key_sha256="sha256:"+"0"*64,
            expected_run_id="research-run",expected_lineage_id="lineage",
            expected_catalog_sha256=catalog,expected_role_registry_sha256=registry_digest())


def test_mutated_signed_manifest_refused(tmp_path):
    path,key,digest,catalog=_signed_grant(tmp_path)
    payload=json.loads(path.read_bytes())
    payload["authorization"]["gpu_sku"]="H200"
    path.write_bytes(canonical_bytes(payload))
    with pytest.raises(Exception):
        SignedResearchComputeGrant.load(path,public_key_hex=key,
            independently_pinned_key_sha256=digest,
            expected_run_id="research-run",expected_lineage_id="lineage",
            expected_catalog_sha256=catalog,expected_role_registry_sha256=registry_digest())


def test_composition_fails_before_s3_inference_for_unmatched_catalog(tmp_path):
    grant=_load(tmp_path)
    class DisallowedS3:
        def get_object(self,**kwargs):
            raise AssertionError("no remote GET during composition")
    index=tmp_path/"catalog.json"
    index.write_bytes(canonical_bytes({"schema":"research-source-catalog.v1","sources":[]}))
    with pytest.raises((PermissionError,ValueError)):
        compose_frozen_diagnosis_research(
            grant=grant,registry_sha256=registry_digest(),
            catalog_path=index,independently_pinned_catalog_sha256="sha256:"+"a"*64,
            state_repository=SQLiteStateRepository(tmp_path/"state.db"),
            artifact_store=FileSystemArtifactStore(tmp_path/"artifacts"),
            s3_read_client=DisallowedS3(),s3_bucket="cviwpryzao",
            original_model_cache=tmp_path/"model-cache",adapter_cache=tmp_path/"adapter-cache",
            eval_pack={},run_id="research-run",lineage_id="lineage")
