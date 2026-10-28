"""CPU internal Judge-entry and Planner owners. No training/certification authority."""
from hephaestus.control.staged_state import StagedOperationResult, StagedOutputRecord
from hephaestus.schemas.diagnosis_contract import DiagnosisReport, EvidenceObservation, DiagnosticHypothesis
from hephaestus.schemas.experiment_contract import InterventionProposal, ExperimentProposal
from hephaestus.policy.internal_research import put_record
from hephaestus.evaluation.cpu_prospective import CpuProspectiveEvaluator


def result(kind, payload, **metadata):
    return StagedOperationResult(records=(StagedOutputRecord(kind, payload),), metadata=metadata)


def with_role(ctx, role, req, base, observation):
    proposal = ctx.role_proposal(role, req, observation)
    return StagedOperationResult(records=(*base.records, StagedOutputRecord("cpu_research_role_proposal", proposal)),
        output_refs=(*base.output_refs, proposal["inference_receipt_ref"]),
        metadata={**base.metadata, "cpu_research_inference_ref": proposal["inference_receipt_ref"],
                  "cpu_role_advisory_only": True, "certified_cuda_role": False})


class CpuResearchEntryRole:
    def __init__(self, context):
        self.ctx = context

    def execute(self, req):
        ctx = self.ctx
        ctx.check_authority()
        if req.substep == "judge_entry_decision":
            payload = {"schema": "cpu-internal-judge-entry.v1", "decision": "bounded_internal_research",
                "profile_ref": ctx.profile_ref, "policy_ref": ctx.admission.policy_ref,
                "external_certification": False}
            return with_role(ctx, "controller", req, result("judge_entry", payload), payload)
        if req.substep == "evidence_collection":
            baseline = CpuProspectiveEvaluator(ctx).generate("entry-baseline")
            ref = put_record(ctx.artifacts, baseline)
            return result("baseline_reference", baseline, baseline_ref=ref, evidence_count=baseline["count"])
        baseline = ctx.source(req, "evidence_collection", "baseline_reference")
        if req.substep == "evidence_based_diagnosis":
            ref = put_record(ctx.artifacts, baseline)
            report = DiagnosisReport("cpu-observed-capability-diagnosis", req.operation_id, req.run_id,
                req.lineage_id, req.stage_name, status="completed",
                observations=[EvidenceObservation("actual-task-score", "actual_pretrained_generation", ref,
                    f"Correct {baseline['correct']}/{baseline['count']}; valid JSON {baseline['json_valid']}/{baseline['count']}.", confidence=1.0)],
                hypotheses=[DiagnosticHypothesis("controlled-task-repair-hypothesis", "inconclusive",
                    "A bounded original-task repair may improve the measured task deficit; mechanism and effect are unknown.",
                    supporting_evidence_refs=[ref], recommended_intervention_kinds=["replace_or_mix_dataset"], confidence=0.25)],
                confidence=0.25, metadata={"measured_task_deficit": baseline["correct"] < baseline["count"],
                    "causal_mechanism": "inconclusive", "baseline_ref": ref, "claim_scope": "internal_prospective"})
            base = result("diagnosis_report", report.to_dict(), diagnosis_report_id=report.report_id,
                missing_evidence=[], blocking_diagnosis=False)
            return with_role(ctx, "diagnosis", req, base, {"correct": baseline["correct"], "count": baseline["count"], "causal_mechanism": "inconclusive"})
        if req.substep != "readiness_to_plan":
            raise ValueError("unsupported CPU entry operation")
        ctx.source(req, "evidence_based_diagnosis", "diagnosis_report")
        return result("planning_readiness", {"ready": True, "scope": "bounded prospective experiment only"}, ready_to_plan=True)


class CpuResearchPlannerRole:
    def __init__(self, context):
        self.ctx = context

    def execute(self, req):
        ctx = self.ctx
        policy = ctx.check_authority()
        diagnosis = ctx.source(req, "evidence_based_diagnosis", "diagnosis_report")
        if req.substep == "diagnosis_handoff":
            return result("diagnosis_handoff", {"diagnosis_report_id": diagnosis["report_id"]})
        if req.substep == "intervention_ranking":
            # Model advice does not choose data rights, training scope or acceptance criteria.
            if not diagnosis["metadata"]["measured_task_deficit"]:
                return StagedOperationResult(status="blocked", blocking_issues=("no_measured_task_deficit",), resumable=False)
            plan = InterventionProposal("bounded-original-task-repair", diagnosis["report_id"], "replace_or_mix_dataset",
                f"Test whether {policy.maximum_training_steps} genuine low-rank optimizer updates on isolated original tasks improve prospective correctness.",
                "training_data", controlled_variables={"model_profile_ref": ctx.profile_ref,
                    "evaluation_protocol_ref": policy.protocol_ref, "optimizer": "AdamW-lr0.0002-lora-r2"},
                expected_cost={"maximum_training_steps": policy.maximum_training_steps},
                success_criteria={"correctness_increases": True, "new_hard_regressions": 0},
                failure_criteria={"nonfinite_gradient": True, "any_new_hard_regression": True},
                rollback_plan="Retain immutable pretrained baseline; no promotion in this internal policy.",
                confidence=0.25, metadata={"hypothesis_not_proven": True, "policy_ref": ctx.admission.policy_ref})
            return with_role(ctx, "planner", req, result("intervention_proposal", plan.to_dict()),
                {"hypothesis": plan.hypothesis, "maximum_updates": policy.maximum_training_steps})
        plan = ctx.source(req, "intervention_ranking", "intervention_proposal")
        if req.substep == "experiment_proposal":
            proposal = ExperimentProposal(policy.experiment_id, req.run_id, req.lineage_id, req.stage_name,
                diagnosis["report_id"], plan["intervention_id"], plan["primary_variable"],
                baseline_ref=diagnosis["metadata"]["baseline_ref"], controlled_variables=plan["controlled_variables"],
                budget={"max_training_steps": ctx.profile.maximum_training_steps, "max_cpu_seconds": ctx.profile.maximum_cpu_seconds},
                success_criteria=plan["success_criteria"], failure_criteria=plan["failure_criteria"],
                required_approvals=["dataset_acquisition", "dataset_selection_approval"],
                rollback_plan=plan["rollback_plan"], metadata={"claim_scope": "internal_prospective", "policy_ref": ctx.admission.policy_ref})
            return result("experiment_proposal", proposal.to_dict(), experiment_id=policy.experiment_id)
        if req.substep == "discovery_request_preparation":
            from hephaestus.data.intelligence import derive_data_requirements
            baseline = ctx.source(req, "evidence_collection", "baseline_reference")
            gaps = sorted({row["family"] for row in baseline["samples"] if not row["correct"]})
            ctx.search_request = derive_data_requirements({"report_id": diagnosis["report_id"],
                "problem_statement": "Repair observed original-task correctness gaps from pinned real CPU baseline generation.",
                "capability_gaps": gaps}, stage=req.stage_name, objective="verifiable_task_repair",
                tokenizer_ref=str(ctx.snapshot / "tokenizer.json"),
                architecture={"context_length": int(ctx.scorer.model.config.max_position_embeddings),
                              "model_id": ctx.profile.model_id, "revision": ctx.profile.model_revision},
                providers=ctx.search_request.provider_allowlist, license_allowlist=policy.allowed_licenses)
            return result("search_request", {"dataset_request": ctx.search_request.to_dict(),
                "model_request": {"model_id": ctx.profile.model_id, "revision": ctx.profile.model_revision}, "discovery_required": True})
        if req.substep != "approval_requirement_discovery":
            raise ValueError("unsupported CPU Planner operation")
        return result("approval_requirements", {"required_approvals": ["dataset_acquisition", "dataset_selection_approval"],
            "policy_authority": ctx.admission.policy_ref}, required_approvals=["dataset_acquisition", "dataset_selection_approval"])
