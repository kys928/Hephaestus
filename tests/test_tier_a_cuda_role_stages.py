"""CPU-only contract tests; mocked outputs are NEVER evidence of actual CUDA."""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from hephaestus.control.staged_state import (
    StagedOperationRequest, StagedOperationResult, StagedOutputRecord,
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
    def __init__(self,store,*,cuda=True,role="diagnosis"):
        self.store=store;self.cuda=cuda;self.role=role
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
            inference_receipt_ref=ref,action_gate=None)


def req(substep="evidence_based_diagnosis"):
    return StagedOperationRequest("workflow","run","lineage","targeted_repair",
        "diagnosis",substep,"native-op",1,(),{})


def stage(tmp_path,*,cuda=True):
    store=FileSystemArtifactStore(tmp_path/"artifacts")
    context=SimpleNamespace(artifacts=store,
        admission=SimpleNamespace(policy_ref="sha256:"+"d"*64))
    reasoner=TestOnlyReasoner(store,cuda=cuda)
    return OriginalCudaResearchObservation(
        "diagnosis","evidence_based_diagnosis",DomainDelegate(),
        context,reasoner),reasoner


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
            object(),SimpleNamespace(gpu_execution_authorized=True))
