"""Deployment wiring for an independently authorized real Diagnostician worker.

This is the same existing SQLite durable queue used for natural-language
objective intake. It NEVER starts a Pod, creates a compute grant, simulates a
role model, promotes weights or merges code. No grant means no live worker.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import os
import re

from hephaestus.jobs import DurableWorker,SQLiteJobQueue
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.sqlite import SQLiteStateRepository
from hephaestus.production.original_cuda_identity import registry_digest
from .durable_jobs import ResearchObjectiveJobHandler
from .live_runtime import SignedResearchComputeGrant,compose_frozen_diagnosis_research

_SHA=re.compile(r"sha256:[0-9a-f]{64}\Z")
_SLUG=re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


@dataclass(frozen=True, slots=True)
class DeployedResearchWorkerScope:
    state_root: Path
    run_id: str
    lineage_id: str
    source_catalog_path: Path
    pinned_source_catalog_sha256: str
    signed_grant_path: Path
    trusted_public_key_path: Path
    pinned_public_key_sha256: str
    trusted_role_registry_sha256: str
    eval_pack_path: Path
    original_model_cache: Path
    original_adapter_cache: Path
    s3_endpoint: str
    s3_region: str
    worker_id: str = "original-frozen-diagnosis-research"

    def validate(self) -> None:
        paths=(self.state_root,self.source_catalog_path,
            self.signed_grant_path,self.trusted_public_key_path,
            self.eval_pack_path,self.original_model_cache,self.original_adapter_cache)
        if (not all(Path(item).is_absolute() and not Path(item).is_symlink()
                    for item in paths)
                or not _SLUG.fullmatch(self.run_id)
                or not _SLUG.fullmatch(self.lineage_id)
                or not _SLUG.fullmatch(self.worker_id)
                or not all(_SHA.fullmatch(value) for value in (
                    self.pinned_source_catalog_sha256,self.pinned_public_key_sha256,
                    self.trusted_role_registry_sha256))
                or self.s3_endpoint!="https://s3api-eu-cz-1.runpod.io/"
                or self.s3_region!="EU-CZ-1"):
            raise PermissionError("research deployment must use independently pinned local, identity and regional scope")
        if self.trusted_role_registry_sha256!=registry_digest():
            raise PermissionError("frozen five-role registry trust root drifted")
        for path in (self.source_catalog_path,self.signed_grant_path,
                     self.trusted_public_key_path,self.eval_pack_path):
            if not path.is_file():
                raise PermissionError("research deployment missing required trusted policy file")


def compose_live_queue_worker(scope: DeployedResearchWorkerScope, *,
                              client=None) -> DurableWorker:
    scope.validate()
    # Read the externally pinned signer key BEFORE creating any model runtime.
    key_hex=scope.trusted_public_key_path.read_text().strip()
    grant=SignedResearchComputeGrant.load(
        scope.signed_grant_path,public_key_hex=key_hex,
        independently_pinned_key_sha256=scope.pinned_public_key_sha256,
        expected_run_id=scope.run_id,expected_lineage_id=scope.lineage_id,
        expected_catalog_sha256=scope.pinned_source_catalog_sha256,
        expected_role_registry_sha256=scope.trusted_role_registry_sha256)
    if (grant.document["region"]!=scope.s3_region
            or grant.document["provider"]!="runpod"):
        raise PermissionError("signed deployment region or provider differs")
    if client is None:
        import boto3
        key=os.environ.get("RUNPOD_S3_ACCESS_KEY_ID","")
        secret=os.environ.get("RUNPOD_S3_SECRET_ACCESS_KEY","")
        if not key or not secret:
            raise PermissionError("read-only external S3 credentials are not configured")
        client=boto3.client(
            "s3",endpoint_url=scope.s3_endpoint,region_name=scope.s3_region,
            aws_access_key_id=key,aws_secret_access_key=secret)
    if not callable(getattr(client,"get_object",None)):
        raise PermissionError("original role and dataset source require GET-only S3 adapter")
    raw=scope.eval_pack_path.read_bytes()
    pack=json.loads(raw)
    if not isinstance(pack,dict):
        raise PermissionError("scientific eval pack is not a JSON object")
    artifacts=FileSystemArtifactStore(scope.state_root/"artifacts")
    repository=SQLiteStateRepository(scope.state_root/"state.db")
    queued=SQLiteJobQueue(scope.state_root/"state.db",initialize=True)
    admitted=compose_frozen_diagnosis_research(
        grant=grant,registry_sha256=scope.trusted_role_registry_sha256,
        catalog_path=scope.source_catalog_path,
        independently_pinned_catalog_sha256=scope.pinned_source_catalog_sha256,
        state_repository=repository,artifact_store=artifacts,
        s3_read_client=client,s3_bucket="cviwpryzao",
        original_model_cache=scope.original_model_cache,
        adapter_cache=scope.original_adapter_cache,
        eval_pack=pack,run_id=scope.run_id,lineage_id=scope.lineage_id)
    handler=ResearchObjectiveJobHandler(artifacts,research=admitted,production_mode=True)
    return DurableWorker(scope.worker_id,queued,handler,
                         lease_seconds=90,heartbeat_interval_seconds=20,
                         poll_interval_seconds=2.0)
