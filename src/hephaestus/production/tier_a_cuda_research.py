"""Exact original generated-task research admission for a CUDA-role campaign.

The original Tier A DATA clearance is reusable only for its byte-identical
source and tokenizer. It never authorizes CUDA spending, protected reference
access, model promotion, or scientific certification.
"""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re

from hephaestus.data.generated_factory import load_authorized_tokens
from hephaestus.training.cpu_chat_json_recipe import exact_chat_records
from hephaestus.storage.verified import canonical_bytes
from hephaestus.production.original_cuda_identity import verified_original_roles, registry_digest

RECIPE = "chat_json_r8_full_epoch_96_v1"
_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class TierACudaDataHandoff:
    """Auditable derived data contract; NOT a CUDA resource authorization."""
    source_policy_ref: str
    source_review_ref: str
    source_materialization_ref: str
    source_tokenizer_sha256: str
    corrected_training_recipe_ref: str
    transformed_training_ref: str
    original_source_record_count: int
    full_epoch_step_count: int
    original_evaluation_protocol_ref: str
    original_cuda_registry_sha256: str
    permitted_scope: str = "internal_generated_task_research_only"
    external_certification: bool = False
    gpu_allocation_authorized: bool = False

    def validate(self):
        digests = (self.source_policy_ref, self.source_review_ref,
            self.source_materialization_ref, self.source_tokenizer_sha256,
            self.corrected_training_recipe_ref, self.transformed_training_ref,
            self.original_evaluation_protocol_ref, self.original_cuda_registry_sha256)
        if (not all(isinstance(x,str) and _SHA.fullmatch(x) for x in digests)
                or self.original_source_record_count != 96
                or self.full_epoch_step_count != 96
                or self.permitted_scope != "internal_generated_task_research_only"
                or self.external_certification is not False
                or self.gpu_allocation_authorized is not False):
            raise PermissionError("Tier A source handoff cannot expand authority or claim execution")
        return self


def serialize_reviewed_supervised_records(records):
    """Canonical JSONL: exactly the bytes hashed by the original chat recipe."""
    return b"".join(canonical_bytes(row) + b"\n" for row in records)


def authenticate_original_tier_a_training(context, *, trusted_registry_sha256):
    """Recheck actual approved rows and exact chat/EOS labels, without training.

    Must be called against the original repository and source artifacts. No
    self-provided source hash or generated test fixture substitutes for the
    original approved bytes.
    """
    if trusted_registry_sha256 != registry_digest():
        raise PermissionError("original CUDA registry trust root changed")
    verified_original_roles(trusted_registry_sha256)
    context.check_authority(for_execution=False)
    policy = context.admission.policy()
    profile = context.profile
    if (profile.training_recipe != RECIPE
            or profile.maximum_training_steps != 96
            or policy.maximum_training_steps != 96
            or context.campaign_config.dataset_seed != 20261028
            or context.campaign_config.include_public is not False
            or context.campaign_config.training_recipe != RECIPE
            or profile.seed != 1117
            or profile.tokenizer_sha256 != policy.tokenizer_sha256
            or context.materialization_ref is None):
        raise PermissionError("GPU data handoff must match original reviewed 96-step research scope")
    rows = load_authorized_tokens(
        context.admission, context.candidate, context.review_ref,
        context.materialization_ref, **context.scope)
    transformed, order, transformed_ref = exact_chat_records(context, rows)
    if (len(transformed) != 96 or len(order) != 96 or
            set(order) != set(range(96))):
        raise ValueError("original corrected training epoch is incomplete")
    from hephaestus.storage.verified import read_verified_json
    frozen = read_verified_json(context.artifacts, policy.protocol_ref)
    if (frozen.get("schema") != "internal-prospective-evaluation.v1"
            or frozen.get("external_certification") is not False
            or frozen.get("historical_reference") is not False):
        raise PermissionError("original evaluation separation cannot be authenticated")
    context.admission.evaluation_store.require_frozen(policy.experiment_id, policy.protocol_ref)
    recipe_file = Path(__file__).resolve().parents[3] / "configs/research/cpu_chat_json_repair_v1.json"
    import json
    recipe_ref = "sha256:" + sha256(canonical_bytes(json.loads(recipe_file.read_bytes()))).hexdigest()
    if context.campaign_config.recipe_sha256 != recipe_ref:
        raise PermissionError("corrected training recipe is not the original reviewed digest")
    # Materialize only training labels, never protected or frozen evaluation text.
    raw = serialize_reviewed_supervised_records(transformed)
    persisted = context.artifacts.put_bytes(raw, media_type="application/jsonl")
    if persisted.artifact_ref != transformed_ref:
        raise PermissionError("new immutable supervised training bytes differ from reviewed transform")
    return TierACudaDataHandoff(
        source_policy_ref=context.admission.policy_ref,
        source_review_ref=context.review_ref,
        source_materialization_ref=context.materialization_ref,
        source_tokenizer_sha256=profile.tokenizer_sha256,
        corrected_training_recipe_ref=recipe_ref,
        transformed_training_ref=transformed_ref,
        original_source_record_count=len(transformed),
        full_epoch_step_count=len(order),
        original_evaluation_protocol_ref=policy.protocol_ref,
        original_cuda_registry_sha256=trusted_registry_sha256,
    ).validate()


@dataclass(slots=True)
class TierACudaRoleDataBridge:
    """Authentic five-role inference dependencies bound to real Tier A data.

    This is a *prepared* nonexecuting bridge, NOT a 45-stage production
    dependency set. Complete GPU stage composition and resource permission
    must be established separately.
    """
    handoff: TierACudaDataHandoff
    reasoner: object
    target_model_id: str
    target_model_revision: str
    gpu_runtime_device: str = "cuda:0"
    gpu_execution_authorized: bool = False
    original_five_role_inferences_completed: bool = False


def prepare_tier_a_original_cuda_role_data_bridge(
        context, *, trusted_registry_sha256, s3_read_client, s3_bucket,
        model_cache_dir, adapter_cache_dir):
    """Connect exact original data/96-step recipe to five original CUDA roles.

    No model weights are loaded; no adapter is downloaded; no GPU is allocated.
    Native Stage owners may consume the returned *reasoner* only after a
    separately scoped GPU execution decision.
    """
    from hephaestus.llm.certified_runtime import (
        CertifiedRoleReasoner, PinnedTransformersRoleGenerator, VerifiedRoleAdapters,
    )
    if not callable(getattr(s3_read_client, "get_object", None)):
        raise PermissionError("original adapter objects require a real read-only S3 client")
    if not isinstance(s3_bucket, str) or not s3_bucket.strip():
        raise PermissionError("original immutable adapter bucket must be explicitly selected")
    handoff = authenticate_original_tier_a_training(
        context, trusted_registry_sha256=trusted_registry_sha256)
    stack, contract = verified_original_roles(trusted_registry_sha256)
    reasoner = CertifiedRoleReasoner(
        stack=stack, contract=contract,
        adapters=VerifiedRoleAdapters(s3_read_client, s3_bucket, Path(adapter_cache_dir)),
        generator=PinnedTransformersRoleGenerator(Path(model_cache_dir), device="cuda:0"),
        artifact_store=context.artifacts)
    if (context.profile.model_id != "HuggingFaceTB/SmolLM2-135M-Instruct"
            or context.profile.model_revision != "12fd25f77366fa6b3b4b768ec3050bf629380bac"):
        raise PermissionError("corrected target model identity differs from original research")
    return TierACudaRoleDataBridge(
        handoff=handoff, reasoner=reasoner,
        target_model_id=context.profile.model_id,
        target_model_revision=context.profile.model_revision,
    )
