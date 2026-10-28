"""Opt-in CPU scientific composition over the UNCHANGED native 45-operation spine.

A separate profile admits genuine CPU research identities. The certified CUDA
composer and its five frozen role identities are not modified or relabeled.
"""
from dataclasses import dataclass
from hephaestus.control.staged_state import StagedAutonomousServices, StagedAutonomousDependencies, StagedOperationResult
from hephaestus.policy.internal_research import put_record
from hephaestus.roles.cpu_research_planning import CpuResearchEntryRole, CpuResearchPlannerRole
from hephaestus.roles.cpu_research_data import CpuResearchDataRole, CpuResearchPreprocessorRole, InternalResearchApprovalService
from hephaestus.roles.cpu_research_execution import CpuResearchTrainingRole, CpuResearchMonitorRole
from hephaestus.roles.cpu_research_evaluation import CpuResearchEvaluatorRole, CpuResearchJudgeRole, CpuResearchActionRole
from hephaestus.production.staged_composition import staged_binding_inventory
from hephaestus.control.spine import SPINE_ORDER
from hephaestus.control.staged_state import PHASE_SUBSTEPS, WORKFLOW_RECORD_COLLECTION
from hephaestus.storage.verified import read_verified_json


@dataclass
class CpuProfiledStage:
    delegate: object
    context: object

    def execute(self, req):
        ctx = self.context
        if (req.workflow_id, req.run_id, req.lineage_id, req.stage_name) != (
                ctx.campaign_config.workflow_id, ctx.campaign_config.run_id,
                ctx.campaign_config.lineage_id, ctx.campaign_config.stage_name):
            raise PermissionError("CPU stage identity differs from authorized campaign")
        ctx.check_authority()
        base = self.delegate.execute(req)
        if base.status != "completed":
            return base
        previous = [r for r in ctx.repository.all(WORKFLOW_RECORD_COLLECTION)
            if r.get("workflow_id") == req.workflow_id and r.get("kind") == "substep_result"
            and r.get("payload", {}).get("completion_marker") is True]
        parent = previous[-1]["artifact_ref"] if previous else None
        ref = put_record(ctx.artifacts, {"schema": "cpu-profiled-domain-operation.v1",
            "workflow_id": req.workflow_id, "run_id": req.run_id,
            "phase": req.phase, "substep": req.substep, "operation_id": req.operation_id,
            "profile_ref": ctx.profile_ref, "policy_ref": ctx.admission.policy_ref,
            "parent_completion_ref": parent, "input_refs": list(req.input_refs),
            "records": [{"kind": r.kind, "payload": r.payload} for r in base.records],
            "external_certification": False})
        return StagedOperationResult(base.status, (*base.output_refs, ref), base.records,
            base.blocking_issues, {**base.metadata, "execution_profile_ref": ctx.profile_ref}, base.resumable)


def compose_cpu_research_dependencies(context):
    context.check_authority()
    if context.state.repository is not context.repository or context.state.artifacts is not context.artifacts:
        raise ValueError("CPU campaign state and artifacts must share original custody")
    def wrap(role):
        return CpuProfiledStage(role(context), context)
    entry, planner = wrap(CpuResearchEntryRole), wrap(CpuResearchPlannerRole)
    data, preprocessor = wrap(CpuResearchDataRole), wrap(CpuResearchPreprocessorRole)
    training, monitor = wrap(CpuResearchTrainingRole), wrap(CpuResearchMonitorRole)
    evaluator, judge, action = wrap(CpuResearchEvaluatorRole), wrap(CpuResearchJudgeRole), wrap(CpuResearchActionRole)
    services = StagedAutonomousServices(judge_entry=entry, evidence_collector=entry,
        diagnosis=entry, plan_readiness=entry, planner=planner,
        dataset_discovery=data, dataset_selection=data, dataset_acquisition=data,
        data_preprocessor=preprocessor, model_discovery=training, model_selection=training,
        training_lifecycle=training, runtime_monitor=monitor, recovery=monitor,
        generation=evaluator, evaluator=evaluator, judge_exit=judge, action_executor=action)
    deps = StagedAutonomousDependencies(services=services, state_repository=context.repository,
        artifact_store=context.artifacts, approval_service=InternalResearchApprovalService(context))
    staged_binding_inventory(deps)
    return deps


def verify_cpu_research_trace(workflow, context):
    from .staged_trace import verify_completed_staged_trace
    trace_ref = verify_completed_staged_trace(workflow, context.repository, context.artifacts)
    context.check_authority(for_execution=False)
    from hephaestus.data.generated_factory import load_authorized_tokens
    load_authorized_tokens(context.admission, context.candidate, context.review_ref,
                           context.materialization_ref, **context.scope)
    rows = [r for r in context.repository.all(WORKFLOW_RECORD_COLLECTION) if r.get("workflow_id") == workflow.workflow_id]
    prior_refs, parent, real_domain, inference_roles = [], None, 0, set()
    for phase in SPINE_ORDER:
        for substep in PHASE_SUBSTEPS[phase]:
            step = workflow.steps[phase.value + "." + substep]
            if step.input_refs != list(dict.fromkeys(prior_refs)):
                raise ValueError("CPU trace parent-state input linkage differs")
            for ref in step.output_refs:
                record = read_verified_json(context.artifacts, ref)
                if record.get("schema") == "cpu-profiled-domain-operation.v1":
                    if record.get("operation_id") != step.operation_id and substep == "replay_evidence" and ref in prior_refs:
                        continue
                    if (record.get("operation_id") != step.operation_id or record.get("workflow_id") != workflow.workflow_id or
                            record.get("profile_ref") != context.profile_ref or record.get("parent_completion_ref") != parent):
                        raise PermissionError("CPU domain operation/profile/parent linkage mismatch")
                    real_domain += 1
            completed = [r for r in rows if r.get("kind") == "substep_result" and r.get("operation_id") == step.operation_id
                         and r.get("payload", {}).get("completion_marker") is True]
            if len(completed) != 1:
                raise ValueError("CPU trace lacks one authentic completion receipt")
            parent = completed[0]["artifact_ref"]
            prior_refs.extend(step.output_refs)
    for row in rows:
        if row.get("kind") == "cpu_research_role_proposal":
            receipt = row["payload"]
            source = read_verified_json(context.artifacts, receipt["inference_receipt_ref"])
            if (source["identity"]["model_id"] != context.profile.model_id or source["identity"]["revision"] != context.profile.model_revision or
                    source["identity"]["model_weights_file_sha256"] != context.profile.snapshot_hashes["model.safetensors"] or
                    source["production_role_certified"] is not False or source["approval_granted"] is not False):
                raise PermissionError("CPU research inference receipt cannot claim CUDA certification")
            inference_roles.add(source["role"])
    if inference_roles != {"controller", "diagnosis", "planner", "evaluator", "judge"} or real_domain != 40:
        raise ValueError("CPU scientific trace lacks genuine domain or five research-role receipts")
    req = type("TraceScope", (), {"workflow_id": workflow.workflow_id, "run_id": workflow.run_id,
        "lineage_id": workflow.lineage_id, "stage_name": workflow.stage_name})()
    training = context.source(req, "lifecycle_launch", "cpu_training_result")
    from pathlib import Path
    from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest
    from hephaestus.training.cpu_research import verify_cpu_checkpoint
    if not verify_cpu_checkpoint(training) or not training["weights_changed"]:
        raise ValueError("CPU trace has no actual optimizer/checkpoint proof")
    return {"schema": "cpu-internal-research-verified-45.v1", "native_trace_ref": trace_ref,
        "profile_ref": context.profile_ref, "policy_ref": context.admission.policy_ref,
        "verified_operations": 45, "verified_domain_operations": real_domain,
        "research_role_inferences": sorted(inference_roles), "actual_training": training,
        "claim_scope": "internal_prospective_generated_task_research",
        "five_frozen_cuda_roles_executed": False, "independent_scientific_certification": False,
        "historical_issue_102_exit_gates_all_satisfied": False}
