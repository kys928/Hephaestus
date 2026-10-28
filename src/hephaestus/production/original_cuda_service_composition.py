"""Fail-closed native GPU composition using only real production services.

Assembly is not an execution receipt. Resource authorization, original rights,
and trusted source approvals are external preconditions, not generated flags.
"""
from dataclasses import dataclass
import re

from hephaestus.control.spine import SPINE_ORDER
from hephaestus.control.staged_state import PHASE_SUBSTEPS
from hephaestus.generation.backends import TransformersCausalLMGenerationBackend
from hephaestus.llm.certified_runtime import (
    CertifiedRoleReasoner, PinnedTransformersRoleGenerator, VerifiedRoleAdapters,
)
from hephaestus.policy.dataset_review import IndependentDatasetReviewPolicy
from hephaestus.production.composition import ProductionRuntime
from hephaestus.production.model_roles import ModelBackedStage
from hephaestus.production.staged_composition import (
    compose_verified_staged_dependencies, staged_binding_inventory,
)
from hephaestus.storage.verified import read_verified_json
from .original_cuda_identity import verified_original_roles


def _is_digest(value):
    return isinstance(value,str) and re.fullmatch(r"sha256:[0-9a-f]{64}",value) is not None


@dataclass(frozen=True)
class OriginalCudaCompositionScope:
    workflow_id: str
    run_id: str
    lineage_id: str
    stage_name: str
    trusted_cpu_archive_sha256: str
    trusted_role_registry_sha256: str
    trusted_original_certificate_sha256: str
    diagnosis_request_ref: str
    baseline_run_artifact_ref: str
    approved_recipe_refs: tuple[str,...]
    approved_baseline_refs: tuple[str,...]
    source_review_refs: tuple[str,...]

    def validate(self):
        for value in (self.workflow_id,self.run_id,self.lineage_id,self.stage_name):
            if not isinstance(value,str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}",value):
                raise PermissionError("original GPU campaign identity invalid")
        digests=(self.trusted_cpu_archive_sha256,self.trusted_role_registry_sha256,
                 self.trusted_original_certificate_sha256,
                 self.diagnosis_request_ref,self.baseline_run_artifact_ref,
                 *self.approved_recipe_refs,*self.source_review_refs)
        if (not self.approved_recipe_refs or not self.approved_baseline_refs or not self.source_review_refs
                or not all(_is_digest(r) for r in digests)
                or len(set(self.source_review_refs))!=len(self.source_review_refs)):
            raise PermissionError("GPU requires independent CPU, recipe, baseline, and source refs")
        return self


def check_cpu_readiness(report, scope):
    """Only recognize a previously independently re-verified CPU archive."""
    expected={"schema":"original-five-cuda-replay-readiness.v1",
              "source_cpu_archive_sha256":scope.trusted_cpu_archive_sha256,
              "original_cuda_role_registry_sha256":scope.trusted_original_certificate_sha256,
              "original_native_completed_operations":45,
              "original_domain_operations":40,
              "actual_optimizer_receipts":96,
              "all_original_source_members_verified":True,
              "gpu_roles_must_execute_same_original_45_operation_graph":True,
              "corrected_target_model_separate_from_cuda_role_stack":True,
              "gpu_run_performed":False,
              "gpu_allocation_authorized":False,
              "external_certification_claimed":False,
              "issue_102_complete":False}
    if any(report.get(k)!=v or type(report.get(k)) is not type(v) for k,v in expected.items()):
        raise PermissionError("CPU evidence is incomplete or wrong scope for original CUDA research")
    if not all(_is_digest(report.get(k)) for k in
               ("native_trace_ref","policy_ref","candidate_checkpoint_sha256")):
        raise PermissionError("original CPU trace/source/checkpoint is uncommitted")
    from .original_cuda_identity import CERTIFICATE, registry_digest
    from hashlib import sha256
    if ("sha256:"+sha256(CERTIFICATE.read_bytes()).hexdigest() !=
            scope.trusted_original_certificate_sha256 or
            scope.trusted_role_registry_sha256 != registry_digest()):
        raise PermissionError("original role registry or certificate trust roots changed")
    stack,_=verified_original_roles(scope.trusted_role_registry_sha256)
    roles=report.get("original_cuda_roles")
    if not isinstance(roles,dict) or set(roles)!=set(stack.roles):
        raise PermissionError("CUDA roles missing from original CPU handoff")
    for role, data in roles.items():
        model=stack.resolve(role)
        if data!={"model_id":model.model_id,"revision":model.revision,
                  "adapter_sha256":model.adapter.sha256}:
            raise PermissionError("CUDA role substituted: "+role)


def compose_original_cuda_science(runtime, *, scope, readiness_ref, approvals,
                                  independent_review, candidate_reviews,
                                  s3_read_client, bucket, model_cache, adapter_cache):
    """Create original five-role scientific services; does NOT invoke run()."""
    scope.validate()
    if type(runtime) is not ProductionRuntime:
        raise TypeError("real production runtime is mandatory")
    if (type(independent_review) is not IndependentDatasetReviewPolicy or
            independent_review.artifacts is not runtime.artifact_store or
            not independent_review.trusted_access_scope_refs):
        raise PermissionError("independently scoped full-corpus source review is required")
    if (set(independent_review.trusted_review_refs)!=set(scope.source_review_refs)
            or set(candidate_reviews.values())!=set(scope.source_review_refs)):
        raise PermissionError("reviewer-accepted original source identities mismatch")
    if (not callable(getattr(approvals,"decision_for",None)) or
            not callable(getattr(s3_read_client,"get_object",None)) or
            not isinstance(bucket,str) or not bucket.strip()):
        raise PermissionError("real scoped approval provider and read-only adapter source needed")
    if not _is_digest(readiness_ref):
        raise PermissionError("authenticated original CPU archive intake required")
    check_cpu_readiness(read_verified_json(runtime.artifact_store,readiness_ref),scope)
    for ref in (scope.diagnosis_request_ref,scope.baseline_run_artifact_ref,
                *scope.approved_recipe_refs,*scope.source_review_refs):
        read_verified_json(runtime.artifact_store,ref)
    if (runtime.data_preprocessor.tokenizer_checker is None or
            runtime.data_preprocessor.record_mapper is None):
        raise PermissionError("source-verified tokenizer and record mapping are mandatory")
    stack,contract=verified_original_roles(scope.trusted_role_registry_sha256)
    reasoner=CertifiedRoleReasoner(
        stack=stack,contract=contract,
        adapters=VerifiedRoleAdapters(s3_read_client,bucket,adapter_cache),
        generator=PinnedTransformersRoleGenerator(model_cache,device="cuda:0"),
        artifact_store=runtime.artifact_store)
    deps=compose_verified_staged_dependencies(
        runtime,approval_service=approvals,reasoner=reasoner,
        generation_backend=TransformersCausalLMGenerationBackend(
            device="cuda",dtype="bfloat16",batch_size=1),
        baseline_run_artifact_ref=scope.baseline_run_artifact_ref,
        approved_recipe_refs=scope.approved_recipe_refs,
        approved_baseline_refs=scope.approved_baseline_refs,
        enable_local_acquisition=False,enable_huggingface_acquisition=True,
        dataset_review_policy=independent_review,
        candidate_review_refs=dict(candidate_reviews),
        enable_training_execution=True,enable_generation_inference=True,
        diagnosis_request_ref=scope.diagnosis_request_ref)
    for role,field in {"controller":"judge_entry","diagnosis":"diagnosis",
                       "planner":"planner","evaluator":"evaluator","judge":"judge_exit"}.items():
        bound=getattr(deps.services,field)
        if type(bound) is not ModelBackedStage or bound.reasoner is not reasoner or bound.role!=role:
            raise PermissionError("frozen CUDA scientific binding missing: "+role)
    bindings=staged_binding_inventory(deps)
    if len(bindings)!=sum(len(PHASE_SUBSTEPS[x]) for x in SPINE_ORDER):
        raise ValueError("original 45-operation service map incomplete")
    return deps,reasoner
