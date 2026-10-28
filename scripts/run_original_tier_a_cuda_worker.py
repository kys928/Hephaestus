#!/usr/bin/env python3
"""Operate one original Tier A 45-operation CUDA campaign on a preallocated pod.

PREPARE is CPU-only and produces an unsigned review request. RUN requires an
independently Ed25519-signed exact authorization plus a *real* pinned CUDA:0
device and the original S3 role adapters. No self-issuance or pod allocation.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

from run_cpu_research_campaign import load_context
from hephaestus.production.original_cuda_identity import registry_digest
from hephaestus.production.signed_cuda_grants import (
    SignedOriginalCudaGrantService, expected_gpu_operation_map,
)
from hephaestus.production.tier_a_cuda_campaign import (
    run_or_resume_original_tier_a_cuda_campaign,
)
from hephaestus.production.tier_a_cuda_research import authenticate_original_tier_a_training
from hephaestus.schemas.cpu_campaign_config import CpuCampaignConfig
from hephaestus.storage.verified import canonical_bytes
from hephaestus.training.tier_a_cuda_target import CudaTargetScope


def prepare_request(ctx, spec, *, maximum_gpu_seconds):
    """Reproducible unsigned request; human signer owns the private key."""
    from hephaestus.production.original_cuda_identity import verified_original_roles
    verified_original_roles(registry_digest())
    handoff=authenticate_original_tier_a_training(
        ctx,trusted_registry_sha256=registry_digest())
    return {
        "schema":"tier-a-original-cuda-compute-authorization.v1",
        "status":"requested", # EXTERNAL operator must approve AND sign.
        "workflow_id":spec.workflow_id, "run_id":spec.run_id,
        "source_policy_ref":handoff.source_policy_ref,
        "reviewed_trainable_ref":handoff.transformed_training_ref,
        "device":"cuda:0","maximum_gpu_seconds":maximum_gpu_seconds,
        "operations":expected_gpu_operation_map(spec.workflow_id),
        "operator_id":"REQUIRES_EXTERNAL_OPERATOR_ID",
        "expires_at":"REQUIRES_UTC_EXPIRY_TIME",
        "external_certification":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--workspace",type=Path,required=True)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--target-model-cache",type=Path,required=True)
    p.add_argument("--original-role-model-cache",type=Path,required=True)
    p.add_argument("--original-adapter-cache",type=Path,required=True)
    p.add_argument("--maximum-gpu-seconds",type=int,default=3600)
    p.add_argument("--prepare-only",action="store_true")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--verify-only",action="store_true")
    p.add_argument("--signed-approval-file",type=Path)
    a=p.parse_args()
    if a.prepare_only and (a.resume or a.verify_only or a.signed_approval_file):
        raise ValueError("prepare-only must not authorize or execute any GPU")
    if not 1<=a.maximum_gpu_seconds<=86400:
        raise ValueError("bounded GPU resource budget required")
    spec=CpuCampaignConfig.from_file(a.config)
    workspace=a.workspace.resolve()
    if not str(workspace).startswith("/workspace/"):
        raise PermissionError("original GPU research requires persistent mounted RunPod volume")
    if a.prepare_only:
        if workspace.exists():
            raise FileExistsError("original GPU campaign must prepare a FRESH workspace")
        ctx,_=load_context(workspace,cache_dir=a.target_model_cache,
                           resume=False,campaign_config_path=a.config)
        request=prepare_request(ctx,spec,maximum_gpu_seconds=a.maximum_gpu_seconds)
        output=workspace/"cuda-grant-unsigned-request.json"
        output.write_bytes(canonical_bytes(request))
        print(json.dumps({"status":"pending_external_signature",
                          "unsigned_request_sha256":"sha256:"+sha256(output.read_bytes()).hexdigest(),
                          "source_policy_ref":ctx.admission.policy_ref,
                          "original_gpu_operations":len(request["operations"]),
                          "gpu_executed":False}))
        return
    if not a.signed_approval_file or not a.signed_approval_file.is_file():
        raise PermissionError("external operator-signed compute authorization is required")
    if not workspace.is_dir():
        raise PermissionError("original preadmitted source workspace not available on persistent storage")
    ctx,existing=load_context(workspace,cache_dir=a.target_model_cache,
                              resume=True,campaign_config_path=a.config)
    external_key=os.environ.get("HEPHAESTUS_TRUSTED_GPU_OPERATOR_PUBLIC_KEY_HEX","")
    trusted_sha=os.environ.get("HEPHAESTUS_TRUSTED_GPU_OPERATOR_PUBLIC_KEY_SHA256","")
    grants=SignedOriginalCudaGrantService.from_signed_file(
        artifacts=ctx.artifacts,path=a.signed_approval_file,
        trusted_public_key_hex=external_key,
        trusted_public_key_sha256=trusted_sha,
        trusted_source_policy_ref=ctx.admission.policy_ref,
        expected_workflow_id=spec.workflow_id,expected_run_id=spec.run_id)
    authorized=grants.authorized
    if (authorized["workflow_id"]!=spec.workflow_id
            or authorized["run_id"]!=spec.run_id
            or authorized.get("reviewed_trainable_ref")!=
                authenticate_original_tier_a_training(
                    ctx,trusted_registry_sha256=registry_digest()).transformed_training_ref):
        raise PermissionError("external compute authorization not bound to original inspected source")
    if a.maximum_gpu_seconds>authorized["maximum_gpu_seconds"]:
        raise PermissionError("worker budget cannot exceed independent operator approval")
    scope=CudaTargetScope(
        workflow_id=spec.workflow_id,run_id=spec.run_id,
        trusted_registry_sha256=registry_digest(),
        trusted_approval_authority=authorized["operator_id"],
        maximum_optimizer_steps=96,
        maximum_gpu_seconds=a.maximum_gpu_seconds,
    )
    if a.verify_only:
        # Verification-only still checks source/grants and complete scientific
        # receipts but never starts a CUDA worker.
        client=object()
    else:
        from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
        from hephaestus.providers.runpod.config import RunPodConfig
        from hephaestus.providers.runpod.storage import RunPodNetworkVolumeStorage
        config=RunPodConfig.from_env()
        client=RunPodNetworkVolumeStorage.from_boto3(
            config,EnvironmentSecretsProvider()).client
    # The original certified role adapter resolver expects boto3 get_object.
    if a.verify_only:
        class NoNetworkRead:
            def get_object(self, **kwargs):
                raise PermissionError("read-only verifier cannot fetch GPU adapters")
        client=NoNetworkRead()
    report=run_or_resume_original_tier_a_cuda_campaign(
        workspace_root=workspace,campaign_config_file=a.config,
        cache_dir=a.target_model_cache,scope=scope,
        gpu_resource_approval_service=grants,
        original_s3_read_client=client,
        original_s3_bucket=os.environ.get("RUNPOD_NETWORK_VOLUME_ID",""),
        original_role_model_cache=a.original_role_model_cache,
        original_adapter_cache=a.original_adapter_cache,
        context_loader=lambda *args,**kwargs:(ctx,existing),
        resume=a.resume,verify_only=a.verify_only)
    print(json.dumps({
        "status":report["status"],"completed_operations":report["completed_operations"],
        "cuda_science_verified":bool(report.get("verified")),
        "historical_external_certification":False,
    },sort_keys=True))


if __name__=="__main__":
    main()
