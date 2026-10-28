"""Post-run authenticity verification of new full native Tier A GPU science.

Never creates missing stage receipts or substitutes role inventories for actual
inference. Evaluate a fresh single campaign; no historical certification.
"""
from hephaestus.control.spine import SPINE_ORDER
from hephaestus.control.staged_state import PHASE_SUBSTEPS, WORKFLOW_RECORD_COLLECTION
from hephaestus.production.staged_trace import verify_completed_staged_trace
from hephaestus.production.original_cuda_identity import verified_original_roles
from hephaestus.storage.verified import read_verified_json
from hephaestus.training.tier_a_cuda_target import verify_cuda_target_checkpoint


def verify_native_tier_a_cuda_trace(workflow, context, *, scope):
    if (not workflow.completion_marker or workflow.status!="completed"
            or workflow.workflow_id!=scope.workflow_id
            or workflow.run_id!=scope.run_id):
        raise PermissionError("original CUDA research native workflow not completely executed")
    native=verify_completed_staged_trace(
        workflow,context.repository,context.artifacts)
    ordered=[phase.value+"."+sub for phase in SPINE_ORDER for sub in PHASE_SUBSTEPS[phase]]
    if len(ordered)!=45 or set(workflow.steps)!=set(ordered):
        raise PermissionError("frozen original 45-operation service graph changed")
    for key in ordered:
        if workflow.steps[key].status!="completed" or not workflow.steps[key].completion_marker:
            raise PermissionError("native CUDA operation uncompleted: "+key)
    rows=[r for r in context.repository.all(WORKFLOW_RECORD_COLLECTION)
          if r.get("workflow_id")==scope.workflow_id]
    role_records=[r["payload"] for r in rows if r.get("kind")=="original_cuda_advisory_role"]
    stack,_=verified_original_roles(scope.trusted_registry_sha256)
    seen=set()
    for row in role_records:
        role=row.get("role")
        if role not in stack.roles or role in seen or row.get("device")!="cuda:0":
            raise PermissionError("original five CUDA role provenance absent or duplicated")
        spec=stack.resolve(role)
        receipt=read_verified_json(context.artifacts,row["inference_receipt_ref"])
        if (row.get("model_id")!=spec.model_id
                or row.get("base_revision")!=spec.revision
                or row.get("adapter_sha256")!=spec.adapter.sha256
                or receipt.get("status")!="accepted"
                or receipt.get("role")!=role
                or receipt.get("model_id")!=spec.model_id
                or receipt.get("base_revision")!=spec.revision
                or receipt.get("adapter_sha256")!=spec.adapter.sha256
                or receipt.get("execution",{}).get("device")!="cuda:0"
                or row.get("advisory_only") is not True
                or row.get("promotion_authorized") is not False):
            raise PermissionError("original CUDA role model/adapter/device receipt mismatch")
        seen.add(role)
    if seen!=set(stack.roles) or len(role_records)!=5:
        raise PermissionError("five actual original CUDA role observations required")
    optimizer=[r for r in context.repository.all("cuda_target_optimizer_steps")
               if r.get("run_id")==scope.run_id]
    if len(optimizer)!=96 or set(r.get("completed_steps") for r in optimizer)!=set(range(1,97)):
        raise PermissionError("all 96 distinct actual original CUDA optimizer receipts required")
    for row in optimizer:
        receipt=read_verified_json(context.artifacts,row["receipt_ref"])
        if (receipt.get("completed_steps")!=row["completed_steps"]
                or receipt.get("execution_device")!="cuda:0"
                or receipt.get("run_id")!=scope.run_id
                or receipt.get("training_recipe","chat_json_r8_full_epoch_96_v1") not in
                    ("chat_json_r8_full_epoch_96_v1",)
                or receipt.get("external_certification") is not False):
            raise PermissionError("original actual CUDA optimizer step identity mismatch")
        verify_cuda_target_checkpoint(receipt)
    request=type("OriginalCUDAReadScope",(),{
        "workflow_id":workflow.workflow_id,"run_id":workflow.run_id,
        "lineage_id":workflow.lineage_id,"stage_name":workflow.stage_name,
    })()
    trained=context.source(request,"lifecycle_launch","cuda_training_result")
    verify_cuda_target_checkpoint(trained)
    if (trained.get("backend_id")!="actual_cuda_transformers_peft"
            or trained.get("completed_optimizer_steps")!=96
            or trained.get("training_recipe")!="chat_json_r8_full_epoch_96_v1"
            or trained.get("full_dataset_epoch_completed") is not True
            or trained.get("model_promoted") is not False):
        raise PermissionError("genuine corrected CUDA target-model training incomplete")
    baseline=context.source(request,"baseline_generation","generation_report")
    candidate=context.source(request,"candidate_generation","generation_report")
    if (baseline.get("execution_device")!="cuda:0"
            or candidate.get("execution_device")!="cuda:0"
            or baseline.get("generation_settings_id")!=candidate.get("generation_settings_id")
            or baseline.get("generation_settings_id")!="greedy-chat64-json-eos-no-truncation.v2"
            or baseline.get("protocol_ref")!=candidate.get("protocol_ref")
            or baseline.get("count")!=candidate.get("count") or baseline.get("count")!=24
            or [s["sample_id"] for s in baseline["samples"]]!=
                [s["sample_id"] for s in candidate["samples"]]):
        raise PermissionError("baseline and candidate must both be real parity-bound CUDA generations")
    from hephaestus.production.tier_a_cuda_research import authenticate_original_tier_a_training
    handoff=authenticate_original_tier_a_training(
        context,trusted_registry_sha256=scope.trusted_registry_sha256)
    if trained["chat_trainable_ref"]!=handoff.transformed_training_ref:
        raise PermissionError("actual CUDA training used wrong original Tier A source bytes")
    return {
        "schema":"tier-a-native-original-five-role-cuda-verified-45.v1",
        "native_trace_ref":native,"native_operations":45,
        "original_five_cuda_role_inferences":sorted(seen),
        "actual_cuda_optimizer_steps":96,
        "cuda_target_model_id":context.profile.model_id,
        "cuda_target_checkpoint_sha256":trained["adapter_sha256"],
        "source_ref":handoff.source_materialization_ref,
        "exact_supervised_training_ref":handoff.transformed_training_ref,
        "evaluation_protocol_ref":baseline["protocol_ref"],
        "correct_before":baseline["correct"],"correct_after":candidate["correct"],
        "valid_json_before":baseline["json_valid"],"valid_json_after":candidate["json_valid"],
        "original_generated_eval_already_seen":True,
        "protected_historical_references_reviewed":False,
        "independent_certification":False,"model_promoted":False,
        "issue_102_exit_criteria_complete":False,
    }
