"""Real, opt-in frozen Diagnostician research composition with signed GPU authority.

A deployment must provide a separately signed, narrowly scoped compute grant,
the exact reviewed source catalog digest, RunPod GET-only S3 client and CUDA
hardware. This module neither provisions Pods nor manufactures authorization.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import re

from hephaestus.llm.certified_runtime import (
    CertifiedRoleReasoner, PinnedTransformersRoleGenerator, VerifiedRoleAdapters,
)
from hephaestus.production.original_cuda_identity import verified_original_roles
from hephaestus.storage.verified import canonical_bytes
from .autonomous_study import (
    CertifiedDiagnosisObserver, EvaluationOnlyResearchJob, ReadOnlyS3SourceReader,
    ResearchObjective, SourceCatalog, VerifiedSourceAcquirer,
)

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


@dataclass(frozen=True, slots=True)
class SignedResearchComputeGrant:
    """Ed25519-reviewed NEW grant; original 45-op CUDA grants are NOT reusable."""

    document: dict[str, object]
    signature_sha256: str
    authority_sha256: str

    @classmethod
    def load(cls, path: Path, *, public_key_hex: str,
             independently_pinned_key_sha256: str,
             expected_run_id: str, expected_lineage_id: str,
             expected_catalog_sha256: str, expected_role_registry_sha256: str
             ) -> "SignedResearchComputeGrant":
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        public_key=bytes.fromhex(public_key_hex)
        if (len(public_key)!=32 or not _DIGEST.fullmatch(independently_pinned_key_sha256)
                or "sha256:"+sha256(public_key).hexdigest()!=independently_pinned_key_sha256):
            raise PermissionError("independently pinned research grant authority does not match")
        signed=json.loads(Path(path).read_bytes())
        if not isinstance(signed,dict) or set(signed)!={"authorization","signature_hex"}:
            raise PermissionError("research compute grant envelope is malformed")
        doc=signed["authorization"]
        if not isinstance(doc,dict):
            raise PermissionError("research authorization must be an object")
        signature=bytes.fromhex(signed["signature_hex"])
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature,canonical_bytes(doc))
        required={
            "schema","status","operator_id","run_id","lineage_id",
            "catalog_sha256","role_registry_sha256","role",
            "provider","region","gpu_sku","device",
            "maximum_inference_calls","maximum_new_tokens",
            "maximum_gpu_seconds","expires_at","approval_scope",
            "training_authorized","code_edit_authorized","model_promotion_authorized",
        }
        if set(doc)!=required:
            raise PermissionError("research grant has extra or missing capability fields")
        try:
            until=datetime.fromisoformat(str(doc["expires_at"]).replace("Z","+00:00"))
        except (TypeError,ValueError) as exc:
            raise PermissionError("research grant expiry malformed") from exc
        if (doc["schema"]!="hephaestus-research-compute-grant.v1"
                or doc["status"]!="approved" or doc["approval_scope"]!="single_evaluation_only_study"
                or doc["role"]!="diagnosis"
                or doc["run_id"]!=expected_run_id
                or doc["lineage_id"]!=expected_lineage_id
                or doc["catalog_sha256"]!=expected_catalog_sha256
                or doc["role_registry_sha256"]!=expected_role_registry_sha256
                or doc["device"]!="cuda:0"
                or doc["provider"]!="runpod"
                or not isinstance(doc["operator_id"],str)
                or doc["operator_id"] in {"","model","self","unknown"}
                or not all(isinstance(doc[k],str) and 1<=len(doc[k])<=100
                       for k in ("region","gpu_sku"))
                or type(doc["maximum_inference_calls"]) is not int
                or doc["maximum_inference_calls"]!=6
                or type(doc["maximum_new_tokens"]) is not int
                or doc["maximum_new_tokens"]!=384
                or type(doc["maximum_gpu_seconds"]) is not int
                or not 1<=doc["maximum_gpu_seconds"]<=3600
                or until.tzinfo is None or until<=datetime.now(timezone.utc)
                or any(doc[k] is not False for k in (
                    "training_authorized","code_edit_authorized",
                    "model_promotion_authorized"))):
            raise PermissionError("signed grant does not authorize this bounded study")
        return cls(dict(doc),"sha256:"+sha256(signature).hexdigest(),
                   independently_pinned_key_sha256)

    def check(self, *, run_id: str, lineage_id: str, calls: int,
              catalog_sha256: str, registry_sha256: str) -> None:
        d=self.document
        expires=datetime.fromisoformat(str(d["expires_at"]).replace("Z","+00:00"))
        if (d["run_id"]!=run_id or d["lineage_id"]!=lineage_id
                or d["catalog_sha256"]!=catalog_sha256
                or d["role_registry_sha256"]!=registry_sha256
                or calls!=6 or calls>d["maximum_inference_calls"]
                or expires<=datetime.now(timezone.utc)):
            raise PermissionError("research GPU grant expired or outside exact study scope")


@dataclass(slots=True)
class AdmittedCudaResearch:
    """The only actual GPU invocation entry in this research composition."""

    job: EvaluationOnlyResearchJob
    grant: SignedResearchComputeGrant
    registry_sha256: str

    def run(self, objective: str, *, run_id: str, lineage_id: str) -> dict[str,object]:
        requested=ResearchObjective.from_text(objective)
        source_pair=self.job.catalog.select(requested)
        count=len(source_pair)*len(requested.seeds)
        self.grant.check(run_id=run_id,lineage_id=lineage_id,calls=count,
                         catalog_sha256=self.job.catalog.catalog_sha256,
                         registry_sha256=self.registry_sha256)
        if type(self.job.observer) is not CertifiedDiagnosisObserver:
            raise PermissionError("no real frozen Diagnostician observer configured")
        # No fake CUDA simulator and no automatic CPU fallback for certified roles.
        import torch
        if not torch.cuda.is_available():
            raise PermissionError("signed CUDA study requires actual CUDA availability")
        from .grant_usage import SingleUseComputeGrantLedger
        ledger=SingleUseComputeGrantLedger(self.job.repository,self.job.artifacts)
        return ledger.run_once(
            grant_signature_sha256=self.grant.signature_sha256,
            run_id=run_id,lineage_id=lineage_id,maximum_calls=count,
            execute=lambda:self.job.run(objective,run_id=run_id,lineage_id=lineage_id))


def compose_frozen_diagnosis_research(*, grant: SignedResearchComputeGrant,
        registry_sha256: str, catalog_path: Path, independently_pinned_catalog_sha256: str,
        state_repository, artifact_store, s3_read_client, s3_bucket: str,
        original_model_cache: Path, adapter_cache: Path, eval_pack: dict,
        run_id: str, lineage_id: str) -> AdmittedCudaResearch:
    """Compose genuine S3 adapter -> original pinned CUDA model -> evidence pilot.

    No model load, remote GET or GPU use occurs during composition. The signed
    grant is still checked before the first actual inference invocation.
    """
    if (not _NAME.fullmatch(run_id) or not _NAME.fullmatch(lineage_id)
            or not _DIGEST.fullmatch(registry_sha256)
            or s3_bucket!="cviwpryzao"
            or not callable(getattr(s3_read_client,"get_object",None))
            or not Path(original_model_cache).is_absolute()
            or not Path(adapter_cache).is_absolute()
            or Path(original_model_cache).is_symlink()
            or Path(adapter_cache).is_symlink()):
        raise PermissionError("real research composition missing trusted identity/source/cache")
    catalog=SourceCatalog.load(catalog_path,
                 independently_pinned_sha256=independently_pinned_catalog_sha256)
    grant.check(run_id=run_id,lineage_id=lineage_id,
                calls=6,catalog_sha256=catalog.catalog_sha256,
                registry_sha256=registry_sha256)
    stack,contract=verified_original_roles(registry_sha256)
    role=stack.resolve("diagnosis")
    if (role.model_id!="mistralai/Ministral-3-3B-Reasoning-2512"
            or role.adapter.sha256!="534d1929d76fc64d07d8d36e85115d5660c113e63b022206bd500d288bdff32c"):
        raise PermissionError("historical Diagnostician original model is not pinned")
    reasoner=CertifiedRoleReasoner(
        stack=stack,contract=contract,
        adapters=VerifiedRoleAdapters(s3_read_client,s3_bucket,Path(adapter_cache)),
        generator=PinnedTransformersRoleGenerator(Path(original_model_cache),device="cuda:0"),
        artifact_store=artifact_store)
    job=EvaluationOnlyResearchJob(
        catalog=catalog,acquirer=VerifiedSourceAcquirer(
            ReadOnlyS3SourceReader(s3_read_client,s3_bucket),
            artifact_store,state_repository),
        observer=CertifiedDiagnosisObserver(reasoner=reasoner),
        repository=state_repository,artifacts=artifact_store,eval_pack=eval_pack)
    return AdmittedCudaResearch(job,grant,registry_sha256)
