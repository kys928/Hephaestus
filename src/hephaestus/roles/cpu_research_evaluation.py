"""CPU prospective Evaluator and Judge owners; promotion/certification stay blocked."""
from pathlib import Path
from hephaestus.control.staged_state import StagedOperationResult, StagedOutputRecord
from hephaestus.control.semantic_judge import SemanticComparisonJudgeAdapter
from hephaestus.evaluation.cpu_prospective import CpuProspectiveEvaluator
from hephaestus.schemas.experiment_contract import ExperimentComparison
from hephaestus.policy.action_registry import evaluate_action_boundary
from hephaestus.policy.internal_research import put_record
from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest
from hephaestus.training.cpu_research import verify_cpu_checkpoint
from hephaestus.storage.verified import read_verified_json
from .cpu_research_planning import result, with_role


class CpuResearchEvaluatorRole:
    def __init__(self, context):
        self.ctx = context

    def execute(self, req):
        ctx = self.ctx
        trained = ctx.source(req, "lifecycle_launch", "cpu_training_result")
        checkpoint = verify_cpu_checkpoint(trained)
        if not validate_checkpoint_manifest(checkpoint)[0]:
            raise ValueError("candidate checkpoint cannot be evaluated")
        service = CpuProspectiveEvaluator(ctx)
        if req.substep == "checkpoint_resolution":
            return result("checkpoint_resolution", trained, checkpoint_ref=str(checkpoint))
        if req.substep == "generation_prompt_materialization":
            protocol = ctx.admission.evaluation_store.require_frozen(ctx.scope["experiment_id"], ctx.profile_protocol)
            return result("generation_prompt_manifest", {"protocol_ref": ctx.profile_protocol,
                "evaluation_ref": protocol["evaluation_ref"], "settings": (
                    "greedy-chat64-json-eos-no-truncation.v2" if
                    ctx.profile.training_recipe == "chat_json_r8_full_epoch_96_v1" else
                    "greedy-max16-exact-prompt-no-truncation.v1")}, eval_pack_id=ctx.profile_protocol)
        if req.substep in {"baseline_generation", "candidate_generation"}:
            generation = (ctx.source(req, "evidence_collection", "baseline_reference") if req.substep == "baseline_generation"
                          else service.generate(req.operation_id, checkpoint=checkpoint))
            return result("generation_report", generation, generation_settings_id=generation["generation_settings_id"])
        baseline = ctx.source(req, "baseline_generation", "generation_report")
        candidate = ctx.source(req, "candidate_generation", "generation_report")
        comparison = service.compare(baseline, candidate)
        if req.substep == "semantic_comparison":
            base = result("experiment_comparison", comparison.to_dict())
            return with_role(ctx, "evaluator", req, base, comparison.effect_summary)
        if req.substep == "deterministic_regression_evidence":
            return result("deterministic_regression_evidence", {"status": comparison.deterministic_gate_status,
                "new_hard_regressions": comparison.effect_summary["new_hard_regressions"],
                "method": "exact_reference_JSON_types_and_values_per_original_task", "cases": baseline["count"]})
        if req.substep == "repeatability_variance_evidence":
            repeated = service.generate(req.operation_id, checkpoint=checkpoint, repeat=1)
            same = [r["output_sha256"] for r in candidate["samples"]] == [r["output_sha256"] for r in repeated["samples"]]
            return result("repeatability_variance_evidence", {"actual_second_generation_ref": put_record(ctx.artifacts, repeated),
                "same_outputs": same, "training_seed_replicates": 1, "training_variance": "unknown",
                "claim_scope": "deterministic_inference_replay_only"})
        if req.substep != "human_review_references":
            raise ValueError("unsupported CPU evaluator operation")
        return result("human_review_references", {"status": "not_performed", "human_attestation_refs": [],
            "comparison_ref": put_record(ctx.artifacts, comparison.to_dict()),
            "required_for_model_promotion": True, "model_promotion_allowed": False,
            "internal_nonpromotion_experiment_permitted": True})


class CpuResearchJudgeRole:
    def __init__(self, context):
        self.ctx = context

    def execute(self, req):
        ctx = self.ctx
        comparison = ExperimentComparison.from_dict(ctx.source(req, "semantic_comparison", "experiment_comparison"))
        trained = ctx.source(req, "lifecycle_launch", "cpu_training_result")
        if req.substep == "governed_verdict":
            verdict = SemanticComparisonJudgeAdapter().decide(comparison, run_id=req.run_id,
                lineage_id=req.lineage_id, candidate_checkpoint_ref=trained["checkpoint_ref"])
            action = verdict.next_action.value
            if action == "promote_checkpoint":
                raise PermissionError("internal CPU research cannot authorize model promotion")
            base = result("judge_exit", verdict.to_dict(), proposed_action=action,
                subject_ref=trained["checkpoint_ref"], certification_state="not_certified")
            return with_role(ctx, "judge", req, base, {"action": action, "comparison_outcome": comparison.primary_outcome})
        if req.substep != "promotion_gate":
            raise ValueError("unsupported CPU Judge operation")
        return result("promotion_gate_report", {"promotion_allowed": False,
            "external_certification": False, "reason": "internal policy is not promotion/certification authority"},
            promotion_allowed=False, action_allowed=False)


class CpuResearchActionRole:
    """Apply only a bounded nonpromotion research decision to append-only state."""
    def __init__(self, context):
        self.ctx = context

    def execute(self, req):
        ctx = self.ctx
        if req.substep != "action_application":
            raise ValueError("unsupported CPU action operation")
        verdict = ctx.source(req, "governed_verdict", "judge_exit")
        action = req.prior_outputs["judge_exit.action_boundary"]["proposed_action"]
        if action not in {"reject_checkpoint", "reject_candidate", "request_recheck", "continue_lineage_best"} or not evaluate_action_boundary(action)["allowed"]:
            raise PermissionError("CPU policy permits only nonpromotion research dispositions")
        payload = {"schema": "cpu-internal-nonpromotion-disposition.v1", "workflow_id": req.workflow_id,
            "run_id": req.run_id, "operation_id": req.operation_id, "action": action,
            "external_certification": False, "model_promoted": False,
            "baseline_retained": True, "effect": "recorded_research_disposition_only"}
        ref = ctx.state.put(req.operation_id, put_record(ctx.artifacts, verdict), payload)
        return result("action_decision", payload, action_applied=True, action_receipt_ref=ref)
