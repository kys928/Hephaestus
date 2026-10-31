"""Composition contract tests only. None performs CUDA inference or GPU launch."""
from dataclasses import replace
from hashlib import sha256

import pytest

from hephaestus.production.original_cuda_identity import (
    CERTIFICATE, verified_original_roles, registry_digest,
)
from hephaestus.production.original_cuda_service_composition import (
    OriginalCudaCompositionScope, check_cpu_readiness, compose_original_cuda_science,
)


def scope():
    return OriginalCudaCompositionScope(
        workflow_id="gpu-source-campaign-1", run_id="gpu-source-run-1",
        lineage_id="frozen-original-roles", stage_name="targeted_repair",
        trusted_cpu_archive_sha256="sha256:"+"a"*64,
        trusted_role_registry_sha256=registry_digest(),
        trusted_original_certificate_sha256="sha256:"+sha256(CERTIFICATE.read_bytes()).hexdigest(),
        diagnosis_request_ref="sha256:"+"b"*64,
        baseline_run_artifact_ref="sha256:"+"c"*64,
        approved_recipe_refs=("sha256:"+"d"*64,),
        approved_baseline_refs=("immutable-baseline",),
        source_review_refs=("sha256:"+"e"*64,),
    ).validate()


def test_original_independent_model_roles_and_frozen_prompt_identity():
    stack,contract=verified_original_roles(registry_digest())
    assert len(stack.roles)==5
    assert len({(r.model_id,r.revision,r.adapter.sha256) for r in stack.roles.values()})==5
    assert stack.automatic_role_dispatch_enabled is False
    assert all(contract.validate_model(r) is None for r in stack.roles.values())


def test_edited_registry_trust_root_is_denied():
    with pytest.raises(PermissionError):
        verified_original_roles("sha256:"+"0"*64)


def test_original_gpu_scope_rejects_missing_source_or_signature_bypass():
    original=scope()
    for kwargs in (
        {"source_review_refs":()},
        {"trusted_cpu_archive_sha256":"latest"},
        {"workflow_id":"../escape"},
        {"approved_recipe_refs":()},
        {"trusted_original_certificate_sha256":"sha256:"+"0"*64},
    ):
        candidate=replace(original,**kwargs)
        if kwargs.get("trusted_original_certificate_sha256"):
            with pytest.raises(PermissionError):
                check_cpu_readiness({},candidate)
        else:
            with pytest.raises(PermissionError):
                candidate.validate()


def test_original_five_cuda_handoff_is_not_an_execution_approval():
    s=scope()
    report={
        "schema":"original-five-cuda-replay-readiness.v1",
        "source_cpu_archive_sha256":s.trusted_cpu_archive_sha256,
        "original_cuda_role_registry_sha256":s.trusted_original_certificate_sha256,
        "original_native_completed_operations":45,
        "original_domain_operations":40,
        "actual_optimizer_receipts":96,
        "all_original_source_members_verified":True,
        "gpu_roles_must_execute_same_original_45_operation_graph":True,
        "corrected_target_model_separate_from_cuda_role_stack":True,
        "gpu_run_performed":False,
        "gpu_allocation_authorized":False,
        "external_certification_claimed":False,
        "issue_102_complete":False,
        "native_trace_ref":"sha256:"+"f"*64,
        "policy_ref":"sha256:"+"1"*64,
        "candidate_checkpoint_sha256":"sha256:"+"2"*64,
        "original_cuda_roles":{
            role:{"model_id":model.model_id,"revision":model.revision,
                  "adapter_sha256":model.adapter.sha256}
            for role,model in verified_original_roles(s.trusted_role_registry_sha256)[0].roles.items()
        },
    }
    # Test-only fabricated input to prove schema validation, NOT science.
    check_cpu_readiness(report,s)
    for field,value in (
        ("gpu_allocation_authorized",True),("actual_optimizer_receipts",2),
        ("original_native_completed_operations",43),("external_certification_claimed",True),
    ):
        with pytest.raises(PermissionError):
            check_cpu_readiness({**report,field:value},s)
    bad={**report,"original_cuda_roles":{**report["original_cuda_roles"],
         "judge":{**report["original_cuda_roles"]["judge"],"model_id":"replacement"}}}
    with pytest.raises(PermissionError):
        check_cpu_readiness(bad,s)


def test_cannot_compose_gpu_without_real_production_runtime():
    with pytest.raises(TypeError):
        compose_original_cuda_science(
            object(),scope=scope(),readiness_ref="sha256:"+"3"*64,
            approvals=None,independent_review=None,candidate_reviews={},
            s3_read_client=None,bucket="",model_cache=None,adapter_cache=None,
        )


def test_native_gpu_spine_rejects_nonproduction_workflow(tmp_path):
    from hephaestus.production.original_cuda_service_composition import build_original_cuda_native_spine
    with pytest.raises(TypeError):
        build_original_cuda_native_spine(object(), scope=scope(), dependencies=object())
