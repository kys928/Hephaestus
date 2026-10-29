"""CPU-only contract tests; mocked outputs are NEVER evidence of actual CUDA."""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from hephaestus.control.staged_state import (
    StagedOperationRequest, StagedOperationResult, StagedOutputRecord,
    StagedApprovalDecision,
)
from hephaestus.production.tier_a_cuda_role_stages import (
    ANCHORS, OriginalCudaResearchObservation,
    prepare_native_tier_a_cuda_role_observations,
)
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.verified import canonical_bytes


class DomainDelegate:
    def execute(self,request):
        return StagedOperationResult(records=(StagedOutputRecord(
            "diagnosis_report",{"measured":True,"correct_before":0}),))


class TestOnlyReasoner:
    """Test doubles cannot pass real composition; only exercise receipt guards."""
    def __init__(self,store,*,cuda=True,role="diagnosis",gate=None):
        self.store=store;self.cuda=cuda;self.role=role;self.gate=gate
        self.calls=[]
    def propose(self,task,registry,*,context):
        self.calls.append((task,registry,context))
        receipt={
            "status":"accepted","role":self.role,"run_id":task.run_id,
            "base_revision":"a"*40,"adapter_sha256":"b"*64,
            "generator":"PinnedTransformersRoleGenerator",
            "execution":{"device":"cuda:0" if self.cuda else "cpu"},
        }
        ref=self.store.put_bytes(canonical_bytes(receipt),media_type="application/json").artifact_ref
        return SimpleNamespace(
            role=self.role,model_id="original-certified-model",
            revision="a"*40,adapter_sha256="b"*64,
            inference_receipt_ref=ref,action_gate=self.gate)


def req(substep="evidence_based_diagnosis"):
    return StagedOperationRequest("workflow","run","lineage","targeted_repair",
        "diagnosis",substep,"native-op",1,(),{})


class TestOnlyGpuGrant:
    """Mocked security gate for logic testing, NEVER a trusted production approver."""
    def __init__(self,store,role):
        self.store,self.role=store,role
    def decision_for(self,request):
        doc={"schema":"original-cuda-resource-grant.v1",
             "workflow_id":request.workflow_id,"run_id":request.run_id,
             "operation_id":request.operation_id,
             "source_policy_ref":request.subject_ref,"role":self.role,
             "device":"cuda:0","status":"approved",
             "authorized_by":"unit-test-fixture-not-a-real-custodian",
             "external_scientific_certification":False}
        ref=self.store.put_bytes(canonical_bytes(doc),media_type="application/json").artifact_ref
        return StagedApprovalDecision(request.request_id,request.operation_id,
            request.subject_ref,"approved",ref,request.requirements)


def stage(tmp_path,*,cuda=True):
    store=FileSystemArtifactStore(tmp_path/"artifacts")
    context=SimpleNamespace(artifacts=store,
        admission=SimpleNamespace(policy_ref="sha256:"+"d"*64))
    reasoner=TestOnlyReasoner(store,cuda=cuda)
    return OriginalCudaResearchObservation(
        "diagnosis","evidence_based_diagnosis",DomainDelegate(),
        context,reasoner,resource_approval_service=TestOnlyGpuGrant(store,"diagnosis")),reasoner


def test_exactly_five_distinct_original_cuda_advisory_entry_points():
    assert set(ANCHORS)=={"controller","diagnosis","planner","evaluator","judge"}
    assert len({field for field,_ in ANCHORS.values()})==5


def test_bound_stage_persists_real_domain_source_before_advisory_request(tmp_path):
    service,reasoner=stage(tmp_path)
    result=service.execute(req())
    assert result.status=="completed"
    assert result.records[0].kind=="diagnosis_report"
    assert result.records[-1].kind=="original_cuda_advisory_role"
    proposal=result.records[-1].payload
    assert proposal["advisory_only"] is True
    assert proposal["cuda_training_executed"] is False
    assert proposal["promotion_authorized"] is False
    assert proposal["independent_certification"] is False
    assert proposal["inference_receipt_ref"] in result.output_refs
    assert len(reasoner.calls)==1


def test_cpu_inference_cannot_masquerade_as_original_cuda_role(tmp_path):
    service,_=stage(tmp_path,cuda=False)
    with pytest.raises(PermissionError,match="real GPU receipt"):
        service.execute(req())


def test_non_role_substep_does_not_generate_model_observation(tmp_path):
    service,reasoner=stage(tmp_path)
    service.execute(req("readiness_to_plan"))
    assert not reasoner.calls


def test_unadmitted_bridge_cannot_bind_native_45_stage_science():
    with pytest.raises(PermissionError):
        prepare_native_tier_a_cuda_role_observations(
            object(),SimpleNamespace(gpu_execution_authorized=True),
            gpu_resource_approval_service=None)


def test_original_cuda_inference_without_distinct_gpu_resource_approval_fails(tmp_path):
    service,reasoner=stage(tmp_path)
    service.resource_approval_service=None
    with pytest.raises(PermissionError,match="resource approval"):
        service.execute(req())
    assert not reasoner.calls


def test_old_tier_a_autoapproval_cannot_be_used_as_gpu_spending_permission(tmp_path):
    service,reasoner=stage(tmp_path)
    class Denied:
        def decision_for(self,request):
            return None
    service.resource_approval_service=Denied()
    with pytest.raises(PermissionError,match="resource grant"):
        service.execute(req())
    assert not reasoner.calls


def test_control_role_veto_does_not_execute_action_or_discard_verified_advice(tmp_path):
    """Mocked inference proves policy separation, NOT any real CUDA success."""
    store=FileSystemArtifactStore(tmp_path/"artifacts")
    context=SimpleNamespace(artifacts=store,
        admission=SimpleNamespace(policy_ref="sha256:"+"d"*64))
    veto=SimpleNamespace(allowed=False,
        failures=("required_verified_fact_not_true:approval_valid",))
    reasoner=TestOnlyReasoner(store,role="controller",gate=veto)
    service=OriginalCudaResearchObservation(
        "controller","judge_entry_decision",DomainDelegate(),
        context,reasoner,resource_approval_service=TestOnlyGpuGrant(store,"controller"))
    request=StagedOperationRequest("workflow","run","lineage","targeted_repair",
        "judge_entry","judge_entry_decision","native-op",1,(),{})
    result=service.execute(request)
    assert result.status=="completed"
    proposal=result.records[-1].payload
    assert proposal["role"]=="controller"
    assert proposal["deterministic_action_allowed"] is False
    assert proposal["effective_action"] is None
    assert proposal["deterministic_action_gate_failures"]==list(veto.failures)
    assert proposal["promotion_authorized"] is False
    assert proposal["advisory_only"] is True
    assert proposal["inference_receipt_ref"] in result.output_refs
    assert result.metadata["original_cuda_effective_action"] is None
    assert len(reasoner.calls)==1


def test_control_role_missing_deterministic_gate_fails_closed(tmp_path):
    store=FileSystemArtifactStore(tmp_path/"artifacts")
    context=SimpleNamespace(artifacts=store,
        admission=SimpleNamespace(policy_ref="sha256:"+"d"*64))
    reasoner=TestOnlyReasoner(store,role="controller")
    service=OriginalCudaResearchObservation(
        "controller","judge_entry_decision",DomainDelegate(),
        context,reasoner,resource_approval_service=TestOnlyGpuGrant(store,"controller"))
    request=StagedOperationRequest("workflow","run","lineage","targeted_repair",
        "judge_entry","judge_entry_decision","native-op",1,(),{})
    with pytest.raises(PermissionError,match="lacks deterministic action gate"):
        service.execute(request)


def test_control_role_denied_resource_grant_is_still_fatal(tmp_path):
    store=FileSystemArtifactStore(tmp_path/"artifacts")
    context=SimpleNamespace(artifacts=store,
        admission=SimpleNamespace(policy_ref="sha256:"+"d"*64))
    veto=SimpleNamespace(allowed=False,failures=("approval_missing",))
    reasoner=TestOnlyReasoner(store,role="controller",gate=veto)
    service=OriginalCudaResearchObservation(
        "controller","judge_entry_decision",DomainDelegate(),
        context,reasoner,resource_approval_service=None)
    request=StagedOperationRequest("workflow","run","lineage","targeted_repair",
        "judge_entry","judge_entry_decision","native-op",1,(),{})
    with pytest.raises(PermissionError,match="resource approval"):
        service.execute(request)
    assert not reasoner.calls
