"""Fail-closed CUDA target contracts: no GPU or trained science is simulated."""
from dataclasses import replace
from types import SimpleNamespace
from hashlib import sha256
import sys

import pytest

from hephaestus.training.tier_a_cuda_target import (
    CudaTargetScope, verified_cuda_grant, require_cuda_device,
)
from hephaestus.control.staged_state import StagedApprovalDecision
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.verified import canonical_bytes

SHA="sha256:"+"a"*64


def scope():
    return CudaTargetScope(
        workflow_id="new-genuine-tier-a-cuda-45",run_id="new-cuda-1117",
        trusted_registry_sha256=SHA,
        trusted_approval_authority="external-user-authorized-compute-owner",
        gpu_device="cuda:0",maximum_optimizer_steps=96,maximum_gpu_seconds=900).validate()


def ctx(tmp_path):
    return SimpleNamespace(
        artifacts=FileSystemArtifactStore(tmp_path/"artifacts"),
        admission=SimpleNamespace(policy_ref=SHA),
        scope={"workflow_id":scope().workflow_id,"run_id":scope().run_id})


class IndependentGrantTestDouble:
    """For refusal/contract unit tests ONLY. No actual authorization."""
    def __init__(self,ctx,purpose,override=None):
        self.ctx=ctx;self.purpose=purpose;self.override=override or {}
    def decision_for(self,request):
        doc={"schema":"original-cuda-resource-grant.v1",
             "workflow_id":request.workflow_id,"run_id":request.run_id,
             "operation_id":request.operation_id,
             "source_policy_ref":request.subject_ref,
             "role":self.purpose,"device":"cuda:0",
             "status":"approved",
             "authorized_by":scope().trusted_approval_authority,
             "maximum_gpu_seconds":scope().maximum_gpu_seconds,
             "external_scientific_certification":False,
             **self.override}
        ref=self.ctx.artifacts.put_bytes(canonical_bytes(doc),
                                        media_type="application/json").artifact_ref
        return StagedApprovalDecision(
            request.request_id,request.operation_id,request.subject_ref,
            "approved",ref,request.requirements)


def test_resource_scope_is_96_steps_cuda_only():
    original=scope()
    assert original.gpu_device=="cuda:0"
    for update in (
        {"maximum_optimizer_steps":8},
        {"gpu_device":"cpu"},
        {"gpu_device":"cuda:1"},
        {"maximum_gpu_seconds":0},
        {"trusted_approval_authority":""},
    ):
        with pytest.raises(PermissionError):
            replace(original,**update).validate()


def test_no_gpu_provider_or_fake_tier_a_autoapproval_grants_compute(tmp_path):
    context=ctx(tmp_path)
    with pytest.raises(PermissionError,match="cannot authorize GPU"):
        verified_cuda_grant(context,"gpu-op-1",scope=scope(),
                            service=None,purpose="target_training")


@pytest.mark.parametrize("override",[
    {"device":"cpu"},
    {"run_id":"other"},
    {"operation_id":"another-op"},
    {"role":"target_evaluation"},
    {"authorized_by":"untrusted-model"},
    {"maximum_gpu_seconds":999999},
    {"external_scientific_certification":True},
])
def test_gpu_grant_rejects_mismatch_before_optimization(tmp_path,override):
    context=ctx(tmp_path)
    with pytest.raises(PermissionError,match="grant|resource"):
        verified_cuda_grant(context,"gpu-op-1",scope=scope(),
             service=IndependentGrantTestDouble(context,"target_training",override),
             purpose="target_training")


def test_pinned_scoped_grant_is_data_access_not_training_completion(tmp_path):
    context=ctx(tmp_path)
    ref=verified_cuda_grant(context,"gpu-op-1",scope=scope(),
            service=IndependentGrantTestDouble(context,"target_training"),
            purpose="target_training")
    assert ref.startswith("sha256:")
    assert len(ref)==71


def test_real_gpu_device_does_not_fallback_to_cpu(monkeypatch):
    fake=SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda:False,device_count=lambda:0))
    monkeypatch.setitem(sys.modules,"torch",fake)
    with pytest.raises(RuntimeError,match="CPU substitution is forbidden"):
        require_cuda_device(scope())


def test_target_evaluator_and_optimizer_are_distinct_cuda_services():
    from hephaestus.training.tier_a_cuda_target import CudaExactTokenTrainer
    from hephaestus.evaluation.tier_a_cuda_target import CudaProspectiveEvaluator
    from hephaestus.roles.tier_a_cuda_science import (
        CudaResearchTrainingRole, CudaResearchEvaluatorRole,
        CudaResearchMonitorRole, CudaResearchEvidenceCollector)
    assert CudaExactTokenTrainer is not CudaProspectiveEvaluator
    assert CudaResearchTrainingRole is not CudaResearchEvaluatorRole
    assert CudaResearchMonitorRole is not CudaResearchEvidenceCollector
