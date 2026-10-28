"""Original CUDA advisory-role bindings on the authentic Tier A research spine.

This bridge preserves the native data/optimizer/Evaluator/Judge service outputs.
The original five GPU roles perform *additional genuine inference* about actual
domain evidence, not substitute fabricated outputs. Their observations are
advisory and cannot approve training, spend funds, or certify performance.

This is a CPU-target / CUDA-role research composition, NOT the fully CUDA
production 45-stage execution required to close issue #102.
"""
from dataclasses import dataclass, replace

from hephaestus.control.deterministic_role_boundary import (
    EvidenceRecord, EvidenceRegistry, DeterministicRoleContext,
)
from hephaestus.control.staged_state import (
    StagedOutputRecord, StagedApprovalRequest, StagedApprovalDecision,
)
from hephaestus.llm.certified_runtime import (
    CertifiedRoleReasoner, PinnedTransformersRoleGenerator,
)
from hephaestus.llm.role_contract import RoleTask, RoleEvidence
from hephaestus.policy.internal_research import put_record
from hephaestus.production.cpu_research import compose_cpu_research_dependencies
from hephaestus.production.original_cuda_identity import registry_digest
from hephaestus.production.staged_composition import staged_binding_inventory
from hephaestus.production.tier_a_cuda_research import (
    TierACudaRoleDataBridge, authenticate_original_tier_a_training,
)
from hephaestus.storage.verified import read_verified_json

ANCHORS = {
    "controller": ("judge_entry", "judge_entry_decision"),
    "diagnosis": ("diagnosis", "evidence_based_diagnosis"),
    "planner": ("planner", "intervention_ranking"),
    "evaluator": ("evaluator", "semantic_comparison"),
    "judge": ("judge_exit", "governed_verdict"),
}


@dataclass(slots=True)
class OriginalCudaResearchObservation:
    """One real role-model proposal attached to a completed native operation."""
    role: str
    substep: str
    delegate: object
    context: object
    reasoner: CertifiedRoleReasoner
    resource_approval_service: object = None

    def _require_resource_grant(self, request):
        # A Tier A right-to-train is NOT an approval to spend GPU resources.
        provider = self.resource_approval_service
        if provider is None or not callable(getattr(provider, "decision_for", None)):
            raise PermissionError("independent GPU resource approval provider missing")
        gate = StagedApprovalRequest(
            request_id="cuda-original:" + request.operation_id,
            workflow_id=request.workflow_id, run_id=request.run_id,
            operation_id=request.operation_id,
            phase="resource_admission", substep="original_cuda_role_inference",
            subject_ref=self.context.admission.policy_ref,
            requirements=("gpu_compute", "original_frozen_cuda_roles"),
        )
        decision = provider.decision_for(gate)
        if (type(decision) is not StagedApprovalDecision
                or decision.request_id != gate.request_id
                or decision.operation_id != gate.operation_id
                or decision.subject_ref != gate.subject_ref
                or decision.status != "approved"
                or set(decision.requirements) != set(gate.requirements)):
            raise PermissionError("no valid independently scoped original CUDA resource grant")
        approved = read_verified_json(self.context.artifacts, decision.approval_ref)
        if (approved.get("schema") != "original-cuda-resource-grant.v1"
                or approved.get("workflow_id") != gate.workflow_id
                or approved.get("run_id") != gate.run_id
                or approved.get("operation_id") != gate.operation_id
                or approved.get("source_policy_ref") != gate.subject_ref
                or approved.get("role") != self.role
                or approved.get("device") != "cuda:0"
                or approved.get("status") != "approved"
                or not approved.get("authorized_by")
                or approved.get("external_scientific_certification") is not False):
            raise PermissionError("verified original CUDA resource approval mismatched operation")
        return decision.approval_ref

    def execute(self, request):
        grant_ref = None
        if request.substep == self.substep:
            grant_ref = self._require_resource_grant(request)
        base = self.delegate.execute(request)
        if base.status != "completed" or request.substep != self.substep:
            return base
        if not base.records:
            raise PermissionError("original CUDA role requires actual domain observation")
        # Freeze the delegate's actual observation before inference; it is
        # not sourced from a candidate model's own unverified annotation.
        source = put_record(self.context.artifacts, {
            "schema": "tier-a-gpu-role-domain-observation.v1",
            "workflow_id": request.workflow_id, "run_id": request.run_id,
            "lineage_id": request.lineage_id, "stage_name": request.stage_name,
            "phase": request.phase, "substep": request.substep,
            "operation_id": request.operation_id,
            "record_kinds": [r.kind for r in base.records],
            "records": [{"kind":r.kind,"payload":r.payload} for r in base.records],
            "source_policy_ref": self.context.admission.policy_ref,
            "claim_scope": "internal_generated_task_research_only",
        })
        registry = EvidenceRegistry({source: EvidenceRecord(
            source, "original_tier_a_verified_domain_observation",
            request.run_id, request.lineage_id,
            sha256=source[7:], observed_sha256=source[7:],
        )})
        observed = self.reasoner.propose(RoleTask(
            role=self.role, situation=(
                "Review the actual bounded original generated-task research "
                "observation. You may advise but cannot approve resources, "
                "promote a checkpoint, or certify historical experiments."),
            evidence=(RoleEvidence(source, "original domain operation evidence"),),
            run_id=request.run_id, lineage_id=request.lineage_id,
            stage_name=request.stage_name,
        ), registry, context=DeterministicRoleContext(
            request.run_id, request.lineage_id, request.stage_name))
        if observed.action_gate is not None and not observed.action_gate.allowed:
            raise PermissionError("original CUDA controller/Judge deterministic action gate rejected output")
        receipt = read_verified_json(self.context.artifacts, observed.inference_receipt_ref)
        if (receipt.get("status") != "accepted"
                or receipt.get("role") != self.role
                or receipt.get("run_id") != request.run_id
                or receipt.get("base_revision") != observed.revision
                or receipt.get("adapter_sha256") != observed.adapter_sha256
                or receipt.get("generator") != "PinnedTransformersRoleGenerator"
                or receipt.get("execution",{}).get("device") != "cuda:0"):
            raise PermissionError("original CUDA inference not proven by real GPU receipt")
        proposal = {
            "schema":"tier-a-original-cuda-advisory-inference.v1",
            "role":self.role, "operation_id":request.operation_id,
            "model_id":observed.model_id, "base_revision":observed.revision,
            "adapter_sha256":observed.adapter_sha256,
            "inference_receipt_ref":observed.inference_receipt_ref,
            "domain_observation_ref":source,
            "gpu_resource_approval_ref":grant_ref,
            "generator":"PinnedTransformersRoleGenerator",
            "device":"cuda:0", "advisory_only":True,
            "cuda_training_executed":False, "promotion_authorized":False,
            "independent_certification":False,
        }
        return replace(base,
            records=(*base.records,StagedOutputRecord("original_cuda_advisory_role",proposal)),
            output_refs=(*base.output_refs,observed.inference_receipt_ref),
            metadata={**base.metadata,"original_cuda_role":self.role,
                      "original_cuda_inference_ref":observed.inference_receipt_ref,
                      "original_cuda_advisory_only":True})


def prepare_native_tier_a_cuda_role_observations(
        context, bridge, *, gpu_resource_approval_service, cuda_target_scope=None):
    """Structurally bind all five original roles; no execution or GPU grant.

    Default preserves CPU-target research for backwards compatibility.
    An explicit CudaTargetScope replaces training/generation/Evaluator/Judge
    with true CUDA execution. Neither scope authorizes historical certification.
    """
    if (type(bridge) is not TierACudaRoleDataBridge
            or type(bridge.reasoner) is not CertifiedRoleReasoner
            or type(bridge.reasoner.generator) is not PinnedTransformersRoleGenerator
            or bridge.reasoner.generator.device != "cuda:0"
            or bridge.reasoner.artifact_store is not context.artifacts
            or bridge.gpu_execution_authorized is not False
            or bridge.original_five_role_inferences_completed is not False):
        raise PermissionError("untrusted or already executed CUDA research bridge")
    fresh = authenticate_original_tier_a_training(
        context, trusted_registry_sha256=registry_digest())
    if fresh != bridge.handoff:
        raise PermissionError("original approved source/recipe changed since CUDA role preparation")
    if (gpu_resource_approval_service is None or
            not callable(getattr(gpu_resource_approval_service,"decision_for",None))):
        raise PermissionError("CUDA research needs a separate scoped resource decision service")
    from hephaestus.roles.cpu_research_data import InternalResearchApprovalService
    if type(gpu_resource_approval_service) is InternalResearchApprovalService:
        raise PermissionError("Tier A dataset autoapproval cannot authorize GPU spend")
    deps = compose_cpu_research_dependencies(context)
    if cuda_target_scope is not None:
        from hephaestus.training.tier_a_cuda_target import CudaTargetScope
        from hephaestus.roles.tier_a_cuda_science import (
            CudaResearchEvidenceCollector, CudaResearchTrainingRole,
            CudaResearchMonitorRole, CudaResearchEvaluatorRole, CudaResearchJudgeRole,
        )
        from hephaestus.production.cpu_research import CpuProfiledStage
        if (type(cuda_target_scope) is not CudaTargetScope
                or cuda_target_scope.workflow_id != context.campaign_config.workflow_id
                or cuda_target_scope.run_id != context.campaign_config.run_id
                or cuda_target_scope.trusted_registry_sha256 != registry_digest()):
            raise PermissionError("CUDA target scope differs from source and frozen role identities")
        cuda_target_scope.validate()
        # Distinct real GPU scientific services, never original CPU optimizer
        # and never a renamed CPU generation receipt.
        gpu_science = {
            "evidence_collector": CudaResearchEvidenceCollector(
                context,scope=cuda_target_scope,grants=gpu_resource_approval_service),
            "training_lifecycle": CudaResearchTrainingRole(
                context,scope=cuda_target_scope,grants=gpu_resource_approval_service,
                original=deps.services.training_lifecycle),
            "runtime_monitor": CudaResearchMonitorRole(context),
            "generation": CudaResearchEvaluatorRole(
                context,scope=cuda_target_scope,grants=gpu_resource_approval_service),
            "evaluator": CudaResearchEvaluatorRole(
                context,scope=cuda_target_scope,grants=gpu_resource_approval_service),
            "judge_exit": CudaResearchJudgeRole(context),
        }
        # The native router uses model_discovery/model_selection and bounded
        # recovery as separate fields, not training_lifecycle/runtime_monitor.
        gpu_science["model_discovery"] = gpu_science["training_lifecycle"]
        gpu_science["model_selection"] = gpu_science["training_lifecycle"]
        gpu_science["recovery"] = gpu_science["runtime_monitor"]
        deps.services = replace(deps.services,**{
            field: CpuProfiledStage(service,context)
            for field,service in gpu_science.items()
        })
    roles = {}
    for role,(field,substep) in ANCHORS.items():
        roles[field] = OriginalCudaResearchObservation(
            role,substep,getattr(deps.services,field),context,bridge.reasoner,
            resource_approval_service=gpu_resource_approval_service)
    deps.services = replace(deps.services,**roles)
    if len(staged_binding_inventory(deps)) != 45:
        raise ValueError("native research stage bindings incomplete")
    return deps
