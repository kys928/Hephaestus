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
from hephaestus.storage.verified import canonical_bytes, read_verified_json


def reopen_original_first_cuda_permission_failure(context, *, scope, grants):
    """Audited retry of ONE unchanged original first-stage permission failure.

    Does not convert an unsuccessful operation into a success or erase its
    previous completion receipt. The first-stage grant is verified again
    before an append-only state transition lets the original engine rerun it.
    All subsequent or repeated failures remain terminal and require review.
    """
    from hashlib import sha256
    from hephaestus.control.staged_state import StagedApprovalRequest
    from hephaestus.production.signed_cuda_grants import SignedOriginalCudaGrantService

    raw=context.repository.get_latest(
        WORKFLOW_STATE_COLLECTION,"workflow_id",scope.workflow_id)
    if raw is None:
        return False
    state=StagedWorkflowState.from_dict(raw)
    if (state.status!="terminal_failure" or state.resumable
            or state.workflow_id!=scope.workflow_id
            or state.run_id!=scope.run_id
            or state.completion_marker):
        return False
    key="judge_entry.judge_entry_decision"
    step=state.steps.get(key)
    if (len(state.steps)!=1 or step is None
            or step.status!="terminal_failure"
            or step.completion_marker
            or step.attempt!=1
            or step.blocking_issues!=["subsystem_exception:PermissionError"]
            or any(v.completion_marker for v in state.steps.values())):
        raise PermissionError("CUDA retry would cross an unreconciled original scientific failure")
    if (type(grants) is not SignedOriginalCudaGrantService
            or grants.authorized.get("workflow_id")!=scope.workflow_id
            or grants.authorized.get("run_id")!=scope.run_id):
        raise PermissionError("fresh original GPU resource authority is required for retried stage")
    request=StagedApprovalRequest(
        request_id="cuda-original:"+step.operation_id,
        workflow_id=scope.workflow_id,run_id=scope.run_id,
        operation_id=step.operation_id,
        phase="resource_admission",substep="original_cuda_role_inference",
        subject_ref=context.admission.policy_ref,
        requirements=("gpu_compute","original_frozen_cuda_roles"))
    decision=grants.decision_for(request)
    if (decision is None or decision.status!="approved"
            or decision.operation_id!=step.operation_id
            or decision.subject_ref!=context.admission.policy_ref):
        raise PermissionError("first original CUDA Controller retry lacks a renewed valid resource grant")
    receipt=read_verified_json(context.artifacts,decision.approval_ref)
    if (receipt.get("role")!="controller"
            or receipt.get("device")!="cuda:0"
            or receipt.get("operation_id")!=step.operation_id
            or receipt.get("source_policy_ref")!=context.admission.policy_ref):
        raise PermissionError("Controller retry grant fails original identity/rights verification")
    # Preserve the terminal failed operation and every old persisted state.
    # Only resume the controller's first operation (attempt 2) with a genuine
    # fresh signed grant: no approval or failed stage is marked completed.
    state.status="interrupted"
    state.resumable=True
    state.blocking_issues=["audited_original_first_cuda_controller_retry"]
    revision=__import__("os").environ.get("HEPHAESTUS_PINNED_EXECUTION_COMMIT","")
    if len(revision)!=40 or any(ch not in "0123456789abcdef" for ch in revision):
        raise PermissionError("CUDA retry must pin its independently inspected Git commit")
    record={"schema":"original-cuda-first-stage-retry-audit.v1",
        "workflow_id":scope.workflow_id,"run_id":scope.run_id,
        "operation_id":step.operation_id,
        "original_terminal_status":"terminal_failure",
        "original_failure_type":"subsystem_exception:PermissionError",
        "prior_attempt":1,"new_attempt":2,
        "retry_code_commit":revision,
        "signed_grant_ref":decision.approval_ref,
        "underlying_source_policy_ref":context.admission.policy_ref,
        "prior_scientific_operations_completed":0,
        "permission_checks_still_enforced":True,
        "independent_historical_certification":False}
    context.repository.append("cuda_first_stage_retry_audit",record,
        operation_id="cuda-audit-"+sha256(canonical_bytes(record)).hexdigest()[:24])
    context.repository.append(WORKFLOW_STATE_COLLECTION,state.to_dict(),
        operation_id="cuda-unblock-"+sha256(canonical_bytes(record)).hexdigest()[:24])
    print("ORIGINAL_CUDA_FIRST_STAGE_RETRY_AUDITED=true",flush=True)
    print("ORIGINAL_CUDA_RETRY_PRIOR_SCIENTIFIC_COMPLETIONS=0",flush=True)
    return True



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
    # Actual optimization and inference require CUDA. An independent read-only
    # verifier may reopen original receipts on CPU without executing science.
    if not verify_only:
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
    if not verify_only:
        from hephaestus.production.authorized_cuda_context import AuthorizedCudaResearchContext
        context=AuthorizedCudaResearchContext.from_signed_approved(
            context,grants=gpu_resource_approval_service,scope=scope)
    if verify_only:
        state=context.repository.get_latest(
            WORKFLOW_STATE_COLLECTION,"workflow_id",scope.workflow_id)
        if state is None:
            raise PermissionError("no original GPU science state exists")
        flow=StagedWorkflowState.from_dict(state)
    else:
        if resume:
            reopen_original_first_cuda_permission_failure(
                context,scope=scope,grants=gpu_resource_approval_service)
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
    from hashlib import sha256
    payload=canonical_bytes(result)
    target=Path(workspace_root)/("cuda-native-report-"+sha256(payload).hexdigest()+".json")
    if not verify_only:
        if target.exists() and target.read_bytes()!=payload:
            raise PermissionError("immutable native GPU research report SHA collision")
        if not target.exists():
            target.write_bytes(payload)
    if not flow.completion_marker:
        raise PermissionError("GPU research remained blocked; incomplete stages are not 45/45")
    return result
