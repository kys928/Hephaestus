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
from hephaestus.control.staged_state import StagedOutputRecord
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

    def execute(self, request):
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


def prepare_native_tier_a_cuda_role_observations(context, bridge):
    """Structurally bind all five original roles; no execution or GPU grant.

    Training and target evaluation remain original CPU research services.
    This helper cannot be used as a substitute for the separate full-CUDA
    production composition required by issue #102.
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
    deps = compose_cpu_research_dependencies(context)
    roles = {}
    for role,(field,substep) in ANCHORS.items():
        roles[field] = OriginalCudaResearchObservation(
            role,substep,getattr(deps.services,field),context,bridge.reasoner)
    deps.services = replace(deps.services,**roles)
    if len(staged_binding_inventory(deps)) != 45:
        raise ValueError("native research stage bindings incomplete")
    return deps
