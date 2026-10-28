"""Opt-in fresh native Tier A CUDA research campaign entry point.

Actual execution only with a separately injected operator-trusted resource
approval provider, existing S3 client and a user-owned persistent workdir.
No GPU allocation, secrets provisioning or certification happens in assembly.
"""
from pathlib import Path

from hephaestus.control.staged_autonomous import build_staged_autonomous_orchestrator
from hephaestus.control.staged_state import WORKFLOW_STATE_COLLECTION,StagedWorkflowState
from hephaestus.llm.cpu_causal_research import MODEL_ID
from hephaestus.production.original_cuda_identity import registry_digest
from hephaestus.production.tier_a_cuda_research import (
    prepare_tier_a_original_cuda_role_data_bridge,
)
from hephaestus.production.tier_a_cuda_role_stages import (
    prepare_native_tier_a_cuda_role_observations,
)
from hephaestus.production.tier_a_cuda_trace import verify_native_tier_a_cuda_trace
from hephaestus.schemas.cpu_campaign_config import CpuCampaignConfig
from hephaestus.training.tier_a_cuda_target import CudaTargetScope, require_cuda_device
from hephaestus.storage.verified import canonical_bytes


def run_or_resume_original_tier_a_cuda_campaign(
        *, workspace_root:Path, campaign_config_file:Path,
        cache_dir:Path, scope:CudaTargetScope, gpu_resource_approval_service,
        original_s3_read_client, original_s3_bucket:str,
        original_role_model_cache:Path, original_adapter_cache:Path,
        context_loader,
        resume:bool=False, verify_only:bool=False):
    """Run authentic native 45 *only if* all source and GPU gates are granted.

    Caller owns GPU pod reservation/teardown and immutably retained storage.
    Offline import/structural checks must not invoke this routine.
    """
    if type(scope) is not CudaTargetScope:
        raise TypeError("independent real CUDA compute authority required")
    scope.validate()
    config=CpuCampaignConfig.from_file(campaign_config_file)
    if (config.schema!="cpu-campaign-config.v2"
            or config.training_recipe!="chat_json_r8_full_epoch_96_v1"
            or config.maximum_training_steps!=96
            or config.dataset_seed!=20261028 or config.cpu_seed!=1117
            or config.include_public is not False
            or (scope.workflow_id,scope.run_id)!=(config.workflow_id,config.run_id)
            or scope.trusted_registry_sha256!=registry_digest()):
        raise PermissionError("native CUDA campaign not bound to original reviewed Tier A scope")
    if (gpu_resource_approval_service is None or
            not callable(getattr(gpu_resource_approval_service,"decision_for",None))
            or not callable(getattr(original_s3_read_client,"get_object",None))):
        raise PermissionError("separate real GPU approval service and original S3 role artifacts required")
    # Explicit runtime gate. No silently slow CPU fallback or fake GPU 45.
    require_cuda_device(scope)
    if not callable(context_loader):
        raise PermissionError("real native original Tier A context loader required")
    context,workspace=context_loader(
        Path(workspace_root).resolve(),cache_dir=Path(cache_dir),
        resume=resume or verify_only,campaign_config_path=campaign_config_file)
    if (context.profile.model_id!=MODEL_ID
            or workspace.workflow_id!=scope.workflow_id
            or workspace.run_id!=scope.run_id):
        raise PermissionError("original CUDA workflow target/runtime drift")
    if verify_only:
        state=context.repository.get_latest(
            WORKFLOW_STATE_COLLECTION,"workflow_id",scope.workflow_id)
        if state is None:
            raise PermissionError("no original GPU science state exists")
        flow=StagedWorkflowState.from_dict(state)
    else:
        bridge=prepare_tier_a_original_cuda_role_data_bridge(
            context,trusted_registry_sha256=scope.trusted_registry_sha256,
            s3_read_client=original_s3_read_client,s3_bucket=original_s3_bucket,
            model_cache_dir=Path(original_role_model_cache),
            adapter_cache_dir=Path(original_adapter_cache))
        deps=prepare_native_tier_a_cuda_role_observations(
            context,bridge,gpu_resource_approval_service=gpu_resource_approval_service,
            cuda_target_scope=scope)
        flow=build_staged_autonomous_orchestrator(
            **workspace.identity,dependencies=deps).run()
    result={"schema":"original-tier-a-cuda-native-campaign-report.v1",
        "workflow_id":flow.workflow_id,"run_id":flow.run_id,
        "status":flow.status,"completed_operations":sum(
            int(step.completion_marker) for step in flow.steps.values()),
        "actual_cuda_role_replays_claimed":False,"external_certification":False}
    if flow.completion_marker:
        result["verified"]=verify_native_tier_a_cuda_trace(
            flow,context,scope=scope)
        result["actual_cuda_role_replays_claimed"]=True
    # Save a truthful state summary whether the original science completed
    # or was safely blocked. Never overwrite an earlier immutable final.
    target=Path(workspace_root)/"cuda_native_campaign_report.json"
    if target.exists() and not resume and not verify_only:
        raise FileExistsError("original CUDA research report already exists")
    if not verify_only:
        target.write_bytes(canonical_bytes(result))
    if not flow.completion_marker:
        raise PermissionError("GPU research remained blocked; incomplete stages are not 45/45")
    return result
