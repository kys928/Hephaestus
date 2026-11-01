"""Evaluator and exit Judge decisions derived from actual model measurements."""
from hephaestus.evaluation.cpu_autonomy import measure
from hephaestus.policy.cpu_autonomy import accept_candidate
from hephaestus.schemas.experiment_contract import ExperimentComparison
from hephaestus.storage.verified import canonical_bytes,read_verified_json
from .cpu_autonomy_planning import result


class CpuAutonomyEvaluatorRole:
    def __init__(self,context): self.ctx=context

    def _measurement(self,req,step):
        c=self.ctx
        return read_verified_json(c.runtime.artifact_store,c.source(req,step,"generation_report")["measurement_ref"])

    def execute(self,req):
        c=self.ctx; c.check(req)
        checkpoint=c.source(req,"checkpoint_resolution","checkpoint_resolution")["checkpoint_ref"]
        if req.substep=="generation_prompt_materialization":
            ref=c.runtime.artifact_store.put_bytes(canonical_bytes({"rows":c.manifest["development"]})).artifact_ref
            return result("generation_prompt_manifest",{"development_ref":ref,"count":8,"test_items_used":False},
                eval_pack_id="public-cpu-development8.v1",generation_settings_id="greedy16-byte-chat-v1")
        if req.substep in {"baseline_generation","candidate_generation"}:
            binding=c.parent() if req.substep=="baseline_generation" else c.binding(checkpoint,req.run_id)
            measured=measure(binding,c.manifest["development"],c.design)
            ref=c.runtime.artifact_store.put_bytes(canonical_bytes(measured)).artifact_ref
            return result("generation_report",{"measurement_ref":ref,"count":8,"model_identity":binding.identity()},
                generation_settings_id="greedy16-byte-chat-v1",eval_pack_id="public-cpu-development8.v1")
        before=self._measurement(req,"baseline_generation")
        after=self._measurement(req,"candidate_generation")
        accepted=accept_candidate(before,after,c.plan.development_nll_margin)
        if req.substep=="semantic_comparison":
            proposal=c.source(req,"experiment_proposal","experiment_proposal")
            comparison=ExperimentComparison("comparison-"+req.run_id,proposal["experiment_id"],
                before["generation"]["model_identity"]["checkpoint_manifest_hash"],candidate_run_ids=[req.run_id],
                primary_outcome="improved" if accepted else "regressed" if after["mean_target_nll"]>before["mean_target_nll"] else "no_change",
                effect_summary={"development_nll_before":before["mean_target_nll"],"development_nll_after":after["mean_target_nll"],
                    "exact_correct_before":before["exact_correct"],"exact_correct_after":after["exact_correct"],"count":8},
                deterministic_gate_status="passed" if accepted else "failed",variance_risk="unknown",
                recommendation="continue_from_checkpoint" if accepted else "reject_checkpoint",
                metadata={"accepted_by_frozen_development_rule":accepted,"checkpoint_ref":checkpoint,
                    "behavioral_gain_demonstrated":False,"external_certification":False})
            return result("experiment_comparison",comparison.to_dict(),comparison_id=comparison.comparison_id)
        if req.substep=="deterministic_regression_evidence":
            return result("deterministic_regression_evidence",{"development_rule_passed":accepted,
                "exact_accuracy_regressed":after["exact_correct"]<before["exact_correct"]},deterministic_passed=accepted)
        if req.substep=="repeatability_variance_evidence":
            return result("repeatability_variance_evidence",{"independent_seed_variance_estimated":False,
                "single_cycle_run_only":True,"cold_fixed_model_reexecution_required":True,"scientific_certification":False})
        if req.substep=="human_review_references":
            return result("human_review_references",{"review_refs":[],"human_review_available":False,"external_certification":False})
        raise ValueError("unsupported CPU evaluation operation")


class CpuAutonomyJudgeRole:
    def __init__(self,context): self.ctx=context

    def execute(self,req):
        c=self.ctx; c.check(req)
        comparison=c.source(req,"semantic_comparison","experiment_comparison")
        accepted=comparison["metadata"]["accepted_by_frozen_development_rule"]
        checkpoint=comparison["metadata"]["checkpoint_ref"]
        action="continue_from_checkpoint" if accepted else "reject_checkpoint"
        if req.substep=="governed_verdict":
            return result("judge_exit",{"proposed_action":action,"checkpoint_ref":checkpoint,"confidence":1.0,
                "confidence_semantics":"deterministic_rule_satisfaction_not_statistical_confidence",
                "stop_after_cycle":c.plan.cycles,"independent_scientific_certification":False},
                proposed_action=action,checkpoint_ref=checkpoint,subject_ref=checkpoint,confidence=1.0)
        if req.substep=="promotion_gate":
            return result("promotion_gate_report",{"promotion_allowed":False,"certification_state":"not_certified",
                "basis":"public bounded CPU benchmark has no production certification authority"},
                promotion_allowed=False,action_allowed=False,certification_state="not_certified")
        raise ValueError("unsupported CPU Judge operation")
