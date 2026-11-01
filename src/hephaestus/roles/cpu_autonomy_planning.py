"""Judge-entry and Planner owners for the finite public CPU experiment."""
from hephaestus.control.staged_state import StagedOperationResult, StagedOutputRecord
from hephaestus.policy.cpu_autonomy import learning_rate
from hephaestus.state.staged_retry_store import StagedRetryStore


def result(kind,payload,**metadata):
    return StagedOperationResult(records=(StagedOutputRecord(kind,payload),),metadata=metadata)


class CpuAutonomyEntryRole:
    def __init__(self,context): self.ctx=context

    def execute(self,req):
        c=self.ctx; c.check(req)
        history=c.runtime.planner.memory_query.intervention_history(req.lineage_id)
        retry=StagedRetryStore(c.runtime.state_repository).current(f"{c.state.program_id}:cycle:{c.cycle}")
        observation={"run_id":req.run_id,"lineage_id":req.lineage_id,"manifest_ref":c.manifest_ref,
            "prior_memory_ids":[m["memory_id"] for m in history],"prior_outcomes":[m["metadata"]["outcome"] for m in history],
            "recovery_decision_ref":retry.decision_ref,"causation_claimed":False}
        if req.substep=="judge_entry_decision":
            return result("judge_entry",{**observation,"decision":"proceed_with_frozen_bounded_cpu_plan","certification_state":"not_certified"})
        if req.substep=="evidence_collection":
            return result("evidence_bundle",observation)
        if req.substep=="evidence_based_diagnosis":
            c.source(req,"evidence_collection","evidence_bundle")
            return result("diagnosis_report",{**observation,"report_id":"diagnosis-"+req.run_id,
                "status":"bounded_diagnostic_hypothesis","hypothesis":("a lower admitted rate may reduce regression" if history and
                    history[-1]["metadata"]["outcome"]!="improved" else "an admitted rate can be tested against the retained parent"),
                "independent_model_judgment":False},missing_evidence=[],blocking_diagnosis=False)
        if req.substep=="readiness_to_plan":
            c.source(req,"evidence_based_diagnosis","diagnosis_report")
            return result("planning_readiness",{"ready":True,"scope":"bounded finite CPU parameter experiment"},ready_to_plan=True)
        raise ValueError("unsupported CPU entry operation")


class CpuAutonomyPlannerRole:
    def __init__(self,context): self.ctx=context

    def execute(self,req):
        c=self.ctx; c.check(req)
        diagnosis=c.source(req,"evidence_based_diagnosis","diagnosis_report")
        memories=c.runtime.planner.memory_query.intervention_history(req.lineage_id)
        rate,reason=learning_rate(c.plan,c.policy,c.cycle,memories)
        if req.substep=="diagnosis_handoff":
            return result("diagnosis_handoff",{"diagnosis_report_id":diagnosis["report_id"]})
        if req.substep=="intervention_ranking":
            return result("intervention_proposal",{"intervention_id":"intervention-"+req.run_id,
                "intervention_kind":"change_training_recipe","primary_variable":"learning_rate",
                "learning_rate":rate,"reason":reason,"required_inputs":[c.manifest_ref],
                "memory_ids":[m["memory_id"] for m in memories],"selection_authority":"frozen_finite_cpu_policy"})
        if req.substep=="experiment_proposal":
            chosen=c.source(req,"intervention_ranking","intervention_proposal")
            proposal=c.proposal(req,chosen["learning_rate"])
            return result("experiment_proposal",proposal.to_dict(),experiment_id=proposal.experiment_id)
        proposal=c.source(req,"experiment_proposal","experiment_proposal")
        if req.substep=="discovery_request_preparation":
            return result("search_request",{"experiment_id":proposal["experiment_id"],"source":"frozen_local_authored_tasks",
                "manifest_ref":c.manifest_ref,"network_allowed":False})
        if req.substep=="approval_requirement_discovery":
            return result("approval_requirements",{"required_approvals":[],"basis":"explicit bounded local CPU campaign task",
                "finite_plan_ref":c.manifest_ref,"paid_resources_allowed":False,"production_promotion_allowed":False})
        raise ValueError("unsupported CPU planner operation")
