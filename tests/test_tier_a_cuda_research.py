"""Fast fail-closed tests for original Tier A -> frozen CUDA research source custody.

These are schema/security tests only; they do not prove any GPU execution.
"""
from dataclasses import replace

import pytest

from hephaestus.production.original_cuda_identity import registry_digest
from hephaestus.production.tier_a_cuda_research import (
    TierACudaDataHandoff, authenticate_original_tier_a_training,
)

S = "sha256:"+"a"*64


def valid():
    return TierACudaDataHandoff(
        source_policy_ref=S, source_review_ref=S, source_materialization_ref=S,
        source_tokenizer_sha256=S, corrected_training_recipe_ref=S,
        transformed_training_ref=S, original_source_record_count=96,
        full_epoch_step_count=96, original_evaluation_protocol_ref=S,
        original_cuda_registry_sha256=registry_digest()).validate()


def test_tier_a_custody_is_not_gpu_execution_or_external_certification():
    admitted = valid()
    assert admitted.permitted_scope=="internal_generated_task_research_only"
    assert admitted.gpu_allocation_authorized is False
    assert admitted.external_certification is False


@pytest.mark.parametrize("key,value", [
    ("source_materialization_ref","main"),
    ("source_review_ref","sha256:"+"z"*64),
    ("original_source_record_count",95),
    ("full_epoch_step_count",8),
    ("gpu_allocation_authorized",True),
    ("external_certification",True),
    ("permitted_scope","historical_protected_references"),
])
def test_original_data_handoff_scope_never_widens(key,value):
    with pytest.raises(PermissionError):
        replace(valid(),**{key:value}).validate()


def test_wrong_independent_original_five_role_registry_denied_first():
    with pytest.raises(PermissionError):
        authenticate_original_tier_a_training(object(),
            trusted_registry_sha256="sha256:"+"0"*64)


def test_prepared_bridge_uses_only_frozen_five_cuda_roles_without_gpu_spend(monkeypatch,tmp_path):
    """Mocked *admission only* to test wiring, not proof of dataset clearance."""
    import hephaestus.production.tier_a_cuda_research as source
    from hephaestus.llm.certified_runtime import (
        CertifiedRoleReasoner, PinnedTransformersRoleGenerator)
    from hephaestus.providers.models.role_stack import REQUIRED_ROLES
    from types import SimpleNamespace
    monkeypatch.setattr(source, "authenticate_original_tier_a_training",
                        lambda ctx, *, trusted_registry_sha256: valid())
    class ReadOnlyAdapter:
        def get_object(self, *, Bucket, Key):
            raise AssertionError("composition must not read adapters")
    store=object()
    ctx=SimpleNamespace(artifacts=store,profile=SimpleNamespace(
        model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        model_revision="12fd25f77366fa6b3b4b768ec3050bf629380bac"))
    bridge=source.prepare_tier_a_original_cuda_role_data_bridge(
        ctx,trusted_registry_sha256=registry_digest(),s3_read_client=ReadOnlyAdapter(),
        s3_bucket="original-scientific-artifacts",
        model_cache_dir=tmp_path/"models",adapter_cache_dir=tmp_path/"adapters")
    assert isinstance(bridge.reasoner,CertifiedRoleReasoner)
    assert type(bridge.reasoner.generator) is PinnedTransformersRoleGenerator
    assert bridge.reasoner.generator.device=="cuda:0"
    assert set(bridge.reasoner.stack.roles)==set(REQUIRED_ROLES)
    assert bridge.reasoner.artifact_store is store
    assert bridge.gpu_execution_authorized is False
    assert bridge.original_five_role_inferences_completed is False
    assert bridge.handoff.external_certification is False
