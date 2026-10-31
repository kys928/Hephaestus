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


def test_native_45_router_uses_true_cuda_target_training_generation_and_judge(
        monkeypatch,tmp_path):
    """Assembly-only test: no actual model, CUDA training or resource allocation."""
    from hephaestus.production import tier_a_cuda_research as handoffmod
    from hephaestus.production import tier_a_cuda_role_stages as stages
    from hephaestus.production.original_cuda_identity import registry_digest
    from hephaestus.production.cpu_research import CpuProfiledStage
    from hephaestus.training.tier_a_cuda_target import CudaTargetScope
    from hephaestus.roles.tier_a_cuda_science import (
        CudaResearchTrainingRole, CudaResearchEvaluatorRole,
        CudaResearchJudgeRole, CudaResearchEvidenceCollector,
    )
    from hephaestus.control.staged_state import (
        StagedAutonomousDependencies,StagedAutonomousServices)
    from hephaestus.storage.filesystem import FileSystemArtifactStore
    # Only construction is mocked; the real runtime must satisfy every source
    # and resource gate when actually executed.
    verified_handoff=SimpleNamespace(source_policy_ref=SHA)
    monkeypatch.setattr(handoffmod,"authenticate_original_tier_a_training",
                        lambda *a,**kw: verified_handoff)
    monkeypatch.setattr(stages,"authenticate_original_tier_a_training",
                        lambda *a,**kw: verified_handoff)
    context=SimpleNamespace(
        artifacts=FileSystemArtifactStore(tmp_path/"artifacts"),
        profile=SimpleNamespace(
            model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            model_revision="12fd25f77366fa6b3b4b768ec3050bf629380bac"),
        campaign_config=SimpleNamespace(
            workflow_id="gpu-scope-native-1",run_id="gpu-scope-run-1"),
    )
    class StubS3:
        def get_object(self, **kwargs):
            raise AssertionError("no S3 fetch during composition")
    bridge=handoffmod.prepare_tier_a_original_cuda_role_data_bridge(
        context,trusted_registry_sha256=registry_digest(),
        s3_read_client=StubS3(),s3_bucket="original-roles",
        model_cache_dir=tmp_path/"model",adapter_cache_dir=tmp_path/"adapters")
    class NullRole:
        def execute(self,req):
            raise AssertionError("no execution during composition")
    names=StagedAutonomousServices.__dataclass_fields__
    delegates=StagedAutonomousServices(**{name:NullRole() for name in names})
    dependencies=StagedAutonomousDependencies(
        services=delegates,state_repository=object(),artifact_store=context.artifacts)
    monkeypatch.setattr(stages,"compose_cpu_research_dependencies",
                        lambda ctx:dependencies)
    monkeypatch.setattr(stages,"staged_binding_inventory",
                        lambda deps:{str(i):"actual-stage" for i in range(45)})
    class ScopedProvider:
        def decision_for(self,request):
            return None
    gpu_scope=CudaTargetScope(
        workflow_id="gpu-scope-native-1",run_id="gpu-scope-run-1",
        trusted_registry_sha256=registry_digest(),
        trusted_approval_authority="operator",maximum_gpu_seconds=900)
    out=stages.prepare_native_tier_a_cuda_role_observations(
        context,bridge,gpu_resource_approval_service=ScopedProvider(),
        cuda_target_scope=gpu_scope)
    assert isinstance(out.services.training_lifecycle.delegate,CudaResearchTrainingRole)
    assert isinstance(out.services.model_selection.delegate,CudaResearchTrainingRole)
    assert isinstance(out.services.evidence_collector.delegate,CudaResearchEvidenceCollector)
    assert isinstance(out.services.runtime_monitor.delegate,
                      __import__("hephaestus.roles.tier_a_cuda_science",fromlist=["CudaResearchMonitorRole"]).CudaResearchMonitorRole)
    assert isinstance(out.services.generation.delegate,CudaResearchEvaluatorRole)
    assert isinstance(out.services.evaluator.delegate.delegate,CudaResearchEvaluatorRole)
    assert isinstance(out.services.judge_exit.delegate.delegate,CudaResearchJudgeRole)
    assert out.services.judge_exit.role=="judge"
    assert out.services.evaluator.role=="evaluator"
    assert out.services.planner.role=="planner"
    assert out.services.diagnosis.role=="diagnosis"
    assert out.services.judge_entry.role=="controller"
