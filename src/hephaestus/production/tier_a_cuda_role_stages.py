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
            print("ORIGINAL_CUDA_ROLE_BOUNDARY=resource_grant_start role="+self.role,flush=True)
            try:
                grant_ref = self._require_resource_grant(request)
            except PermissionError:
                print("ORIGINAL_CUDA_ROLE_PERMISSION_DENIAL=resource_grant role="+self.role,flush=True)
                raise
            print("ORIGINAL_CUDA_ROLE_BOUNDARY=resource_grant_approved role="+self.role,flush=True)
        if request.substep == self.substep:
            print("ORIGINAL_CUDA_ROLE_BOUNDARY=domain_delegate_start role="+self.role,flush=True)
        try:
            base = self.delegate.execute(request)
        except PermissionError:
            print("ORIGINAL_CUDA_ROLE_PERMISSION_DENIAL=domain_delegate role="+self.role,flush=True)
            raise
        if request.substep == self.substep:
            print("ORIGINAL_CUDA_ROLE_BOUNDARY=domain_delegate_returned role="+self.role+" status="+base.status,flush=True)
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
        print("ORIGINAL_CUDA_ROLE_BOUNDARY=certified_inference_start role="+self.role,flush=True)
        task = RoleTask(
            role=self.role, situation=(
                "Review the actual bounded original generated-task research "
                "observation. You may advise but cannot approve resources, "
                "promote a checkpoint, or certify historical experiments."),
            evidence=(RoleEvidence(source, "original domain operation evidence"),),
            run_id=request.run_id, lineage_id=request.lineage_id,
            stage_name=request.stage_name,
        )
        role_context = DeterministicRoleContext(
            request.run_id, request.lineage_id, request.stage_name)
        rejected_format_refs = []
        try:
            observed = self.reasoner.propose(task, registry, context=role_context)
        except PermissionError:
            print("ORIGINAL_CUDA_ROLE_PERMISSION_DENIAL=certified_inference role="+self.role,flush=True)
            raise
        except ValueError as exc:
            # The genuine r5 Planner and r8 Evaluator both produced a
            # frozen-schema evidence citation rejection. Restrict one new
            # GPU inference to exactly those source-bound original boundaries.
            # One new GPU inference may be requested with exact VERIFIED
            # source ID feedback; NEVER alter the generated output or insert
            # a citation ourselves. On any further parser error fail closed.
            citation_role_substeps = {
                "planner": "intervention_ranking",
                "evaluator": "semantic_comparison",
            }
            if (self.role in citation_role_substeps
                    and request.substep == citation_role_substeps[self.role]
                    and str(exc) == "role response cites missing, duplicate or empty evidence"):
                rejected_ref = getattr(exc, "rejected_inference_receipt_ref", None)
                if not isinstance(rejected_ref, str) or not rejected_ref.startswith("sha256:"):
                    raise PermissionError("original citation rejection lacks immutable receipt") from exc
                rejected = read_verified_json(self.context.artifacts, rejected_ref)
                if (rejected.get("schema") != "role-inference.v1"
                        or rejected.get("status") != "rejected"
                        or rejected.get("error") != str(exc)
                        or rejected.get("role") != self.role
                        or rejected.get("run_id") != request.run_id
                        or rejected.get("lineage_id") != request.lineage_id
                        or rejected.get("stage_name") != request.stage_name
                        or rejected.get("evidence_refs") != [source]
                        or rejected.get("generator") != "PinnedTransformersRoleGenerator"
                        or rejected.get("execution",{}).get("device") != "cuda:0"):
                    raise PermissionError("original rejected GPU citation inference evidence invalid") from exc
                if (len(task.evidence) != 1
                        or task.evidence[0].ref != source
                        or not source.startswith("sha256:")):
                    raise PermissionError("citation retry needs exactly one verified source") from exc
                rejected_format_refs.append(rejected_ref)
                print("ORIGINAL_CUDA_ROLE_BOUNDARY=bounded_citation_repair_start role="+self.role,flush=True)
                repair_task=replace(task,situation=task.situation + (
                    "\\nFORMAT-ONLY RETRY: The previous JSON cited absent, repeated "
                    "or empty evidence IDs. The ONE independently verified "
                    "domain evidence identifier for this case is: " + source +
                    ". Use the identical seven-key frozen JSON schema. "
                    "If the observation genuinely supports your assessment, "
                    "the evidence_refs JSON array must contain exactly this "
                    "identifier ONCE. Do not invent any source, claim, "
                    "training result, approval, or scientific fact. "
                    "If the evidence does not support an intervention, "
                    "choose a conservative assessment from the original "
                    "role vocabulary instead."))
                try:
                    observed=self.reasoner.propose(
                        repair_task,registry,context=role_context)
                except ValueError:
                    print("ORIGINAL_CUDA_ROLE_CITATION_REPAIR=failed_closed role="+self.role,flush=True)
                    raise
                print("ORIGINAL_CUDA_ROLE_CITATION_REPAIR=accepted role="+self.role,flush=True)
            else:
                # This is a single narrow repair for the observed genuine r2
                # Controller failure. The frozen original model uses greedy
                # decoding: repeating the identical prompt would reproduce its
                # invalid primary_variable. Never parse or patch model text here.
                if (self.role != "controller"
                        or request.substep != "judge_entry_decision"
                        or str(exc) != "role response has invalid primary_variable"):
                    raise
                prior_ref = getattr(exc, "rejected_inference_receipt_ref", None)
                if not isinstance(prior_ref, str):
                    raise PermissionError("original Controller format rejection lacks immutable receipt") from exc
                prior = read_verified_json(self.context.artifacts, prior_ref)
                if (prior.get("schema") != "role-inference.v1"
                        or prior.get("status") != "rejected"
                        or prior.get("error") != str(exc)
                        or prior.get("role") != "controller"
                        or prior.get("run_id") != request.run_id
                        or prior.get("lineage_id") != request.lineage_id
                        or prior.get("stage_name") != request.stage_name
                        or prior.get("evidence_refs") != [source]
                        or prior.get("generator") != "PinnedTransformersRoleGenerator"
                        or prior.get("execution", {}).get("device") != "cuda:0"):
                    raise PermissionError("original Controller rejected CUDA inference evidence invalid") from exc
                vocab = self.reasoner.contract.profile("controller")["vocabulary"]["primary_variable"]
                if (not isinstance(vocab, list) or not vocab
                        or any(not isinstance(value, str) or not value.isidentifier()
                               for value in vocab)
                        or len(set(vocab)) != len(vocab)):
                    raise PermissionError("frozen Controller primary_variable vocabulary invalid") from exc
                rejected_format_refs.append(prior_ref)
                print("ORIGINAL_CUDA_ROLE_BOUNDARY=bounded_format_repair_start role=controller", flush=True)
                # The original frozen contract renders the prompt for both real
                # attempts. Only this format feedback changes; evidence, role,
                # policy, decoding and semantic task remain fixed. No action runs.
                repair_task = replace(task, situation=task.situation + (
                    "\\nFORMAT-ONLY RETRY: The prior response had an out-of-vocabulary "
                    "primary_variable. Do not invent checkpoint integrity, approvals "
                    "or scientific facts. Produce ONE new JSON response with the same "
                    "seven frozen keys and evidence IDs. The primary_variable MUST "
                    "be exactly one of: " + ", ".join(vocab) + "."))
                try:
                    observed = self.reasoner.propose(
                        repair_task, registry, context=role_context)
                except ValueError:
                    print("ORIGINAL_CUDA_ROLE_FORMAT_REPAIR=failed_closed role=controller", flush=True)
                    raise
                print("ORIGINAL_CUDA_ROLE_FORMAT_REPAIR=accepted role=controller", flush=True)
        print("ORIGINAL_CUDA_ROLE_BOUNDARY=certified_inference_returned role="+self.role,flush=True)
        # Controller/Judge outputs are advice, never authority. A veto blocks
        # the proposed action; it does not invalidate the real inference
        # evidence or the independently completed domain observation.
        gate = observed.action_gate
        if self.role in {"controller","judge"} and gate is None:
            raise PermissionError("original CUDA control role requires deterministic action gate")
        action_allowed = gate is None or gate.allowed
        if not action_allowed:
            print("ORIGINAL_CUDA_ROLE_BOUNDARY=advisory_action_vetoed role="+self.role,flush=True)
        print("ORIGINAL_CUDA_ROLE_BOUNDARY=receipt_verification_start role="+self.role,flush=True)
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
            "deterministic_action_allowed":action_allowed,
            "deterministic_action_gate_failures":list(gate.failures) if gate else [],
            "effective_action":None,  # No model action is dispatched by this role.
            "bounded_format_repair_attempts":len(rejected_format_refs),
            "rejected_inference_receipt_refs":list(rejected_format_refs),
            "cuda_training_executed":False, "promotion_authorized":False,
            "independent_certification":False,
        }
        return replace(base,
            records=(*base.records,StagedOutputRecord("original_cuda_advisory_role",proposal)),
            output_refs=(*base.output_refs,observed.inference_receipt_ref),
            metadata={**base.metadata,"original_cuda_role":self.role,
                      "original_cuda_inference_ref":observed.inference_receipt_ref,
                      "original_cuda_advisory_only":True,
                      "original_cuda_action_allowed":action_allowed,
                      "original_cuda_effective_action":None,
                      "original_cuda_bounded_format_repair_attempts":len(rejected_format_refs)})


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
                context,scope=cuda_target_scope,grants=gpu_resource_approval_service,
                original_role_runtime=bridge.reasoner.generator),
            "training_lifecycle": CudaResearchTrainingRole(
                context,scope=cuda_target_scope,grants=gpu_resource_approval_service,
                original=deps.services.training_lifecycle,
                original_role_runtime=bridge.reasoner.generator),
            "runtime_monitor": CudaResearchMonitorRole(context),
            "generation": CudaResearchEvaluatorRole(
                context,scope=cuda_target_scope,grants=gpu_resource_approval_service,
                original_role_runtime=bridge.reasoner.generator),
            "evaluator": CudaResearchEvaluatorRole(
                context,scope=cuda_target_scope,grants=gpu_resource_approval_service,
                original_role_runtime=bridge.reasoner.generator),
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
