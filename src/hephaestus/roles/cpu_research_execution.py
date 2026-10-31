"""CPU training engineer and runtime monitor owners, with real optimizer/checkpoint IO."""
from pathlib import Path
from hephaestus.training.cpu_research import CpuExactTokenTrainer
from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest
from hephaestus.training.cpu_research import verify_cpu_checkpoint
from hephaestus.schemas.experiment_contract import TrainingRunHandle
from hephaestus.policy.internal_research import put_record
from hephaestus.data.generated_factory import load_authorized_tokens
from .cpu_research_planning import result


class CpuResearchTrainingRole:
    def __init__(self, context):
        self.ctx = context

    def execute(self, req):
        ctx = self.ctx
        policy = ctx.check_authority()
        if req.substep == "model_discovery":
            return result("model_discovery_result", {"status": "discovered", "model_id": ctx.profile.model_id,
                "revision": ctx.profile.model_revision, "snapshot_hashes": ctx.profile.snapshot_hashes,
                "model_source": "actual_local_pinned_pretrained_snapshot", "role_adapter": "none"}, model_revision=ctx.profile.model_revision)
        if req.substep == "model_selection":
            discovered = ctx.source(req, "model_discovery", "model_discovery_result")
            return result("model_selection_decision", {"status": "selected", "model": discovered,
                "profile_ref": ctx.profile_ref, "authority": "preauthorized_bounded_cpu_profile",
                "required_approvals": []}, subject_ref=ctx.profile_ref, required_approvals=[])
        ctx.source(req, "model_selection", "model_selection_decision")
        if req.substep == "training_input_binding":
            data = ctx.source(req, "trainable_data_contract", "trainable_data_contract")
            rows = load_authorized_tokens(ctx.admission, ctx.candidate, ctx.review_ref, ctx.materialization_ref, **ctx.scope)
            ctx.admission.evaluation_store.record_exposure(policy.experiment_id, policy.protocol_ref, ctx.materialization_ref)
            return result("training_input_binding", {"materialization_ref": ctx.materialization_ref,
                "profile_ref": ctx.profile_ref, "policy_ref": ctx.admission.policy_ref, "rows": len(rows),
                "maximum_width": data["maximum_width"]})
        if req.substep == "launch_configuration":
            ctx.source(req, "training_input_binding", "training_input_binding")
            return result("training_configuration", {"model_profile_ref": ctx.profile_ref,
                "materialization_ref": ctx.materialization_ref, "steps": ctx.profile.maximum_training_steps,
                "learning_rate": 0.0005 if ctx.profile.training_recipe == "chat_json_r8_full_epoch_96_v1" else 0.0002,
                "optimizer": "AdamW", "lora_rank": 8 if ctx.profile.training_recipe == "chat_json_r8_full_epoch_96_v1" else 2,
                "lora_alpha": 16 if ctx.profile.training_recipe == "chat_json_r8_full_epoch_96_v1" else 4,
                "lora_dropout": 0.0, "target_modules": ["q_proj", "v_proj"],
                "context_policy": "no_truncation", "label_policy": ("chat_prefix_masked_with_eos" if
                    ctx.profile.training_recipe == "chat_json_r8_full_epoch_96_v1" else "exact_prompt_mask_minus100"),
                "sample_order": ("interleaved_four_families_one_full_epoch" if
                    ctx.profile.training_recipe == "chat_json_r8_full_epoch_96_v1" else "source_order_modulo_rows"),
                "training_recipe": ctx.profile.training_recipe, "seed": ctx.profile.seed}, config_identity=ctx.profile_ref, seed_identity=str(ctx.profile.seed))
        if req.substep == "lifecycle_launch":
            ctx.source(req, "launch_configuration", "training_configuration")
            trained = CpuExactTokenTrainer(ctx).run(req.operation_id)
        elif req.substep == "training_status_poll":
            trained = ctx.source(req, "lifecycle_launch", "cpu_training_result")
        else:
            raise ValueError("unsupported CPU training operation")
        checkpoint = verify_cpu_checkpoint(trained)
        if not validate_checkpoint_manifest(checkpoint)[0]:
            raise ValueError("CPU training checkpoint invalid")
        metrics_ref = put_record(ctx.artifacts, trained)
        handle = TrainingRunHandle(req.run_id, policy.experiment_id, "actual_cpu_transformers_peft",
            "completed", checkpoint_refs=[str(checkpoint)], metrics_ref=metrics_ref,
            metadata={"profile_ref": ctx.profile_ref, "actual_optimizer_steps": trained["completed_optimizer_steps"]})
        base = result("training_run_handle", handle.to_dict(), training_status="completed",
            run_id=req.run_id, experiment_id=policy.experiment_id, checkpoint_ref=str(checkpoint))
        from hephaestus.control.staged_state import StagedOperationResult, StagedOutputRecord
        return StagedOperationResult(records=(*base.records, StagedOutputRecord("cpu_training_result", trained)), metadata=base.metadata)


class CpuResearchMonitorRole:
    def __init__(self, context):
        self.ctx = context

    def execute(self, req):
        ctx = self.ctx
        ctx.check_deadline()
        trained = ctx.source(req, "training_status_poll", "cpu_training_result")
        if not verify_cpu_checkpoint(trained) or not trained["weights_changed"]:
            raise ValueError("CPU monitor found invalid optimizer/checkpoint evidence")
        if req.substep == "runtime_observation":
            return result("runtime_observation", {"training_status": "completed", "losses": trained["losses"],
                "checkpoint_ref": trained["checkpoint_ref"], "completed_steps": trained["completed_optimizer_steps"]})
        if req.substep == "runtime_evidence":
            ctx.source(req, "runtime_observation", "runtime_observation")
            return result("runtime_evidence", {"run_id": req.run_id, "experiment_id": ctx.scope["experiment_id"],
                "observed_status": "completed", "checkpoint_ref": trained["checkpoint_ref"], "incidents": [],
                "metrics_ref": put_record(ctx.artifacts, trained)}, monitor_outcome="healthy")
        evidence = ctx.source(req, "runtime_evidence", "runtime_evidence")
        if req.substep == "bounded_recovery_advice":
            return result("recovery_decision", {"status": "not_required", "basis": evidence,
                "recovery_executed": False})
        if req.substep != "runtime_control_governance":
            raise ValueError("unsupported CPU monitor operation")
        return result("runtime_control_recommendation", {"recommendation": "no_intervention_completed_training",
            "checkpoint_ref": trained["checkpoint_ref"], "control_action_executed": False})
