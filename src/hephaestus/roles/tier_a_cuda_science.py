"""Tier A CUDA target training/evaluation native-stage owners, no CPU fallback.

These replace only the target-science operations. The original five certified
CUDA *role* models remain separate and continue through their own bindings.
All other native 45 workflow phase owners retain original Tier A governance.
"""
from hephaestus.control.staged_state import StagedOperationResult, StagedOutputRecord
from hephaestus.control.semantic_judge import SemanticComparisonJudgeAdapter
from hephaestus.evaluation.tier_a_cuda_target import CudaProspectiveEvaluator
from hephaestus.policy.internal_research import put_record
from hephaestus.roles.cpu_research_planning import result, with_role
from hephaestus.schemas.experiment_contract import TrainingRunHandle, ExperimentComparison
from hephaestus.training.tier_a_cuda_target import (
    CudaExactTokenTrainer, verify_cuda_target_checkpoint,
)
from hephaestus.data.generated_factory import load_authorized_tokens


class CudaResearchEvidenceCollector:
    def __init__(self,context, *, scope, grants):
        self.ctx=context;self.eval=CudaProspectiveEvaluator(context,scope=scope,grants=grants)

    def execute(self,request):
        if request.substep!="evidence_collection":
            raise ValueError("CUDA evidence collector must only generate actual CUDA baseline")
        baseline=self.eval.generate(request.operation_id)
        source=put_record(self.ctx.artifacts,baseline)
        return result("baseline_reference",baseline,baseline_ref=source,evidence_count=baseline["count"])


class CudaResearchTrainingRole:
    def __init__(self,context, *, scope, grants, original):
        self.ctx=context;self.scope=scope;self.grants=grants;self.original=original

    def execute(self,req):
        ctx=self.ctx
        if req.substep=="model_discovery":
            return result("model_discovery_result",{
                "status":"discovered","model_id":ctx.profile.model_id,
                "revision":ctx.profile.model_revision,
                "snapshot_hashes":ctx.profile.snapshot_hashes,
                "model_source":"verified_original_pinned_pretrained_snapshot",
                "target_training_device":"cuda:0","role_adapter":"none",
            },model_revision=ctx.profile.model_revision)
        if req.substep=="model_selection":
            discovered=ctx.source(req,"model_discovery","model_discovery_result")
            return result("model_selection_decision",{
                "status":"selected","model":discovered,
                "profile_ref":ctx.profile_ref,
                "authority":"Tier_A_data_only_separate_GPU_resource_grant_required",
                "required_approvals":[],
            },subject_ref=ctx.profile_ref,required_approvals=[])
        if req.substep=="training_input_binding":
            source=ctx.source(req,"trainable_data_contract","trainable_data_contract")
            from hephaestus.production.tier_a_cuda_research import authenticate_original_tier_a_training
            handoff=authenticate_original_tier_a_training(
                ctx,trusted_registry_sha256=self.scope.trusted_registry_sha256)
            ctx.admission.evaluation_store.record_exposure(
                ctx.scope["experiment_id"],ctx.profile_protocol,ctx.materialization_ref)
            return result("training_input_binding",{
                "materialization_ref":ctx.materialization_ref,
                "corrected_chat_trainable_ref":handoff.transformed_training_ref,
                "policy_ref":ctx.admission.policy_ref,"profile_ref":ctx.profile_ref,
                "source_rows_admitted":96,"training_steps":96,
                "device":"cuda:0","maximum_width":source["maximum_width"],
                "GPU_execution_permission":"separately_verified_at_lifecycle_launch",
            })
        if req.substep=="launch_configuration":
            bound=ctx.source(req,"training_input_binding","training_input_binding")
            return result("training_configuration",{
                "profile_ref":ctx.profile_ref,
                "source_materialization_ref":ctx.materialization_ref,
                "corrected_chat_trainable_ref":bound["corrected_chat_trainable_ref"],
                "optimizer":"AdamW","learning_rate":0.0005,
                "steps":96,"lora_rank":8,"lora_alpha":16,
                "lora_dropout":0.0,"target_modules":["q_proj","v_proj"],
                "sample_order":"four_families_interleaved_full_original_epoch",
                "labels":"exact_chat_prefix_mask_minus100_target_eos",
                "device":"cuda:0","training_execution_authorized":False,
                "external_certification":False,
            },config_identity=ctx.profile_ref,seed_identity=str(ctx.profile.seed))
        if req.substep=="lifecycle_launch":
            conf=ctx.source(req,"launch_configuration","training_configuration")
            if conf["steps"]!=96 or conf["device"]!="cuda:0":
                raise PermissionError("original CUDA launch config changed")
            trained=CudaExactTokenTrainer(ctx,scope=self.scope,grants=self.grants).run(req.operation_id)
        elif req.substep=="training_status_poll":
            trained=ctx.source(req,"lifecycle_launch","cuda_training_result")
        else:
            raise ValueError("CUDA training role only accepts original model/train operations")
        checkpoint=verify_cuda_target_checkpoint(trained)
        metrics_ref=put_record(ctx.artifacts,trained)
        handle=TrainingRunHandle(req.run_id,ctx.scope["experiment_id"],
            "actual_cuda_transformers_peft","completed",
            checkpoint_refs=[str(checkpoint)],metrics_ref=metrics_ref,
            metadata={"device":"cuda:0","optimizer_steps":96,
                      "gpu_resource_approval_ref":trained["gpu_resource_approval_ref"]})
        base=result("training_run_handle",handle.to_dict(),training_status="completed",
            run_id=req.run_id,experiment_id=ctx.scope["experiment_id"],
            checkpoint_ref=str(checkpoint))
        return StagedOperationResult(records=(*base.records,
            StagedOutputRecord("cuda_training_result",trained)),metadata=base.metadata)


class CudaResearchMonitorRole:
    def __init__(self,context):
        self.ctx=context

    def execute(self,req):
        ctx=self.ctx
        trained=ctx.source(req,"training_status_poll","cuda_training_result")
        verify_cuda_target_checkpoint(trained)
        if not trained["weights_changed"] or trained["completed_optimizer_steps"]!=96:
            raise PermissionError("actual CUDA optimizer proof incomplete")
        if req.substep=="runtime_observation":
            return result("runtime_observation",{
                "training_status":"completed","execution_device":"cuda:0",
                "checkpoint_ref":trained["checkpoint_ref"],
                "optimizer_steps":trained["completed_optimizer_steps"],
                "optimizer_receipt_source_ref":trained["chat_trainable_ref"]})
        if req.substep=="runtime_evidence":
            obs=ctx.source(req,"runtime_observation","runtime_observation")
            return result("runtime_evidence",{
                "run_id":req.run_id,"experiment_id":ctx.scope["experiment_id"],
                "observed_status":"completed","execution_device":"cuda:0",
                "checkpoint_ref":trained["checkpoint_ref"],"incidents":[],
                "metrics_ref":put_record(ctx.artifacts,trained)},monitor_outcome="healthy")
        if req.substep=="bounded_recovery_advice":
            source=ctx.source(req,"runtime_evidence","runtime_evidence")
            return result("recovery_decision",{"status":"not_required","basis":source,
                                               "recovery_executed":False})
        if req.substep=="runtime_control_governance":
            ctx.source(req,"runtime_evidence","runtime_evidence")
            return result("runtime_control_recommendation",{
                "recommendation":"no_intervention_completed_training",
                "checkpoint_ref":trained["checkpoint_ref"],
                "control_action_executed":False})
        raise ValueError("unsupported CUDA monitor substep")


class CudaResearchEvaluatorRole:
    def __init__(self,context, *, scope, grants):
        self.ctx=context
        self.eval=CudaProspectiveEvaluator(context,scope=scope,grants=grants)

    def execute(self,req):
        ctx=self.ctx
        trained=ctx.source(req,"lifecycle_launch","cuda_training_result")
        checkpoint=verify_cuda_target_checkpoint(trained)
        if req.substep=="checkpoint_resolution":
            return result("checkpoint_resolution",trained,checkpoint_ref=str(checkpoint))
        if req.substep=="generation_prompt_materialization":
            policy=ctx.admission.evaluation_store.require_frozen(
                ctx.scope["experiment_id"],ctx.profile_protocol)
            return result("generation_prompt_manifest",{
                "protocol_ref":ctx.profile_protocol,
                "evaluation_ref":policy["evaluation_ref"],
                "settings":"greedy-chat64-json-eos-no-truncation.v2",
                "execution_device":"cuda:0","reused_generated_evaluation":True,
            },eval_pack_id=ctx.profile_protocol)
        if req.substep in ("baseline_generation","candidate_generation"):
            data=(ctx.source(req,"evidence_collection","baseline_reference")
                  if req.substep=="baseline_generation" else
                  self.eval.generate(req.operation_id,checkpoint=checkpoint))
            if data.get("execution_device")!="cuda:0":
                raise PermissionError("candidate/baseline CUDA generation cannot use CPU")
            return result("generation_report",data,
                generation_settings_id=data["generation_settings_id"])
        baseline=ctx.source(req,"baseline_generation","generation_report")
        candidate=ctx.source(req,"candidate_generation","generation_report")
        comparison=self.eval.compare(baseline,candidate)
        if req.substep=="semantic_comparison":
            return with_role(ctx,"evaluator",req,
                result("experiment_comparison",comparison.to_dict()),
                comparison.effect_summary)
        if req.substep=="deterministic_regression_evidence":
            return result("deterministic_regression_evidence",{
                "status":comparison.deterministic_gate_status,
                "new_hard_regressions":comparison.effect_summary["new_hard_regressions"],
                "method":"exact_original_JSON_types_values_cuda_greedy",
                "cases":baseline["count"]})
        if req.substep=="repeatability_variance_evidence":
            replay=self.eval.generate(req.operation_id,checkpoint=checkpoint,repeat=1)
            same=[r["output_sha256"] for r in candidate["samples"]]==[
                r["output_sha256"] for r in replay["samples"]]
            return result("repeatability_variance_evidence",{
                "actual_cuda_generation_ref":put_record(ctx.artifacts,replay),
                "same_outputs":same,"training_seed_replicates":1,
                "training_variance":"unknown",
                "claim_scope":"deterministic_cuda_inference_replay_only"})
        if req.substep=="human_review_references":
            return result("human_review_references",{
                "status":"not_performed","human_attestation_refs":[],
                "comparison_ref":put_record(ctx.artifacts,comparison.to_dict()),
                "required_for_model_promotion":True,
                "model_promotion_allowed":False,
                "internal_nonpromotion_experiment_permitted":True})
        raise ValueError("unsupported native CUDA evaluator substep")


class CudaResearchJudgeRole:
    def __init__(self,context):
        self.ctx=context

    def execute(self,req):
        ctx=self.ctx
        comparison=ExperimentComparison.from_dict(
            ctx.source(req,"semantic_comparison","experiment_comparison"))
        trained=ctx.source(req,"lifecycle_launch","cuda_training_result")
        if req.substep=="governed_verdict":
            verdict=SemanticComparisonJudgeAdapter().decide(
                comparison,run_id=req.run_id,lineage_id=req.lineage_id,
                candidate_checkpoint_ref=trained["checkpoint_ref"])
            action=verdict.next_action.value
            if action=="promote_checkpoint":
                raise PermissionError("Tier A CUDA research does not authorize promotion")
            return with_role(ctx,"judge",req,result("judge_exit",verdict.to_dict(),
                proposed_action=action,subject_ref=trained["checkpoint_ref"],
                certification_state="not_certified"),
                {"action":action,"comparison_outcome":comparison.primary_outcome})
        if req.substep=="promotion_gate":
            return result("promotion_gate_report",{
                "promotion_allowed":False,"external_certification":False,
                "reason":"internal generated-task CUDA research is not independent certification"
            },promotion_allowed=False,action_allowed=False)
        raise ValueError("unsupported CUDA Judge substep")
