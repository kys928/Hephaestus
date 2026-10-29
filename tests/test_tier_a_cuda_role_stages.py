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


def test_historical_controller_checkpoint_action_veto_remains_nonexecuting_advice(tmp_path):
    """Reproduce the exact three failed facts from original preserved inference.

    The genuine CUDA grant, inference and stage outcome are NOT simulated here:
    test doubles exercise boundary routing only; GPU science remains unproven.
    """
    from hephaestus.control.deterministic_role_boundary import guard_role_output
    store=FileSystemArtifactStore(tmp_path/"artifacts")
    context=SimpleNamespace(artifacts=store,
        admission=SimpleNamespace(policy_ref="sha256:"+"d"*64))
    gate_holder=[]

    class HistoricalActionReasoner(TestOnlyReasoner):
        def propose(self,task,registry,*,context):
            observed=super().propose(task,registry,context=context)
            gate=guard_role_output(role="controller",
                output={"action":"continue_from_checkpoint",
                    "evidence_refs":[x.ref for x in task.evidence]},
                context=context,evidence_registry=registry)
            gate_holder.append(gate)
            observed.action_gate=gate
            return observed

    reasoner=HistoricalActionReasoner(store,role="controller")
    service=OriginalCudaResearchObservation(
        "controller","judge_entry_decision",DomainDelegate(),
        context,reasoner,
        resource_approval_service=TestOnlyGpuGrant(store,"controller"))
    request=StagedOperationRequest(
        "workflow","run","lineage","targeted_repair",
        "judge_entry","judge_entry_decision","native-op",1,(),{})
    result=service.execute(request)
    gate=gate_holder[0]
    assert gate.allowed is False
    assert set(gate.failures)=={
        "required_verified_fact_not_true:candidate_checkpoint_integrity_valid",
        "required_verified_fact_not_true:candidate_checkpoint_lineage_match",
        "required_verified_fact_not_true:provenance_valid",
    }
    assert result.status=="completed"
    proposal=result.records[-1].payload
    assert proposal["advisory_only"] is True
    assert proposal["deterministic_action_allowed"] is False
    assert set(proposal["deterministic_action_gate_failures"])==set(gate.failures)
    assert proposal["effective_action"] is None
    assert proposal["promotion_authorized"] is False
    assert result.metadata["original_cuda_effective_action"] is None
    assert len(reasoner.calls)==1


def test_controller_missing_deterministic_gate_still_denied(tmp_path):
    store=FileSystemArtifactStore(tmp_path/"artifacts")
    context=SimpleNamespace(artifacts=store,
        admission=SimpleNamespace(policy_ref="sha256:"+"d"*64))
    reasoner=TestOnlyReasoner(store,role="controller")
    service=OriginalCudaResearchObservation(
        "controller","judge_entry_decision",DomainDelegate(),
        context,reasoner,
        resource_approval_service=TestOnlyGpuGrant(store,"controller"))
    request=StagedOperationRequest(
        "workflow","run","lineage","targeted_repair",
        "judge_entry","judge_entry_decision","native-op",1,(),{})
    with pytest.raises(PermissionError,match="requires deterministic action gate"):
        service.execute(request)


def test_controller_action_veto_never_overrides_denied_resource_authority(tmp_path):
    store=FileSystemArtifactStore(tmp_path/"artifacts")
    context=SimpleNamespace(artifacts=store,
        admission=SimpleNamespace(policy_ref="sha256:"+"d"*64))
    reasoner=TestOnlyReasoner(store,role="controller")
    service=OriginalCudaResearchObservation(
        "controller","judge_entry_decision",DomainDelegate(),
        context,reasoner,
        resource_approval_service=None)
    request=StagedOperationRequest(
        "workflow","run","lineage","targeted_repair",
        "judge_entry","judge_entry_decision","native-op",1,(),{})
    with pytest.raises(PermissionError,match="resource approval provider missing"):
        service.execute(request)
    assert not reasoner.calls


def _controller_rejected_format_fixture(tmp_path, *, second_attempt="accepted"):
    """No GPU: use REAL frozen parser + mock reasoner to exercise failure contract."""
    import json
    from pathlib import Path
    from hephaestus.llm.role_contract import RoleMasteryContract
    store=FileSystemArtifactStore(tmp_path/"artifacts")
    context=SimpleNamespace(artifacts=store,
        admission=SimpleNamespace(policy_ref="sha256:"+"d"*64))
    contract=RoleMasteryContract.load(Path(__file__).resolve().parents[1]
        /"configs/models/hephaestus_role_mastery_runtime_v1.json")
    attempted=[]

    class FormatRejectedThenRetried(TestOnlyReasoner):
        def __init__(self):
            super().__init__(store,role="controller")
            self.contract=contract
        def propose(self,task,registry,*,context):
            attempted.append(task)
            if len(attempted)==1 or second_attempt=="rejected":
                row={
                    "decision":"blocked","action":"request_recheck",
                    "primary_variable":"not_in_frozen_controller_vocabulary",
                    "confidence":0.5,
                    "evidence_refs":[task.evidence[0].ref],
                    "uncertainties":[],
                    "rationale":"Scientific evidence requires another bounded review."}
                try:
                    self.contract.parse("controller",json.dumps(row),
                                        {e.ref for e in task.evidence})
                except ValueError as err:
                    assert str(err)=="role response has invalid primary_variable"
                    rejected={
                        "schema":"role-inference.v1","status":"rejected",
                        "error":str(err),"role":"controller",
                        "run_id":task.run_id,"lineage_id":task.lineage_id,
                        "stage_name":task.stage_name,
                        "evidence_refs":[e.ref for e in task.evidence],
                        "generator":"PinnedTransformersRoleGenerator",
                        "execution":{"device":"cuda:0","synthetic_test_double":True},
                        "raw_output":"fixture-only-invalid-primary-variable"}
                    ref=store.put_bytes(canonical_bytes(rejected),
                        media_type="application/json").artifact_ref
                    err.rejected_inference_receipt_ref=ref
                    raise
            result=super().propose(task,registry,context=context)
            result.action_gate=SimpleNamespace(
                allowed=False,
                failures=("required_verified_fact_not_true:provenance_valid",))
            return result

    reasoner=FormatRejectedThenRetried()
    service=OriginalCudaResearchObservation(
        "controller","judge_entry_decision",DomainDelegate(),
        context,reasoner,
        resource_approval_service=TestOnlyGpuGrant(store,"controller"))
    request=StagedOperationRequest("workflow","run","lineage","targeted_repair",
        "judge_entry","judge_entry_decision","native-op",1,(),{})
    return service,request,attempted,store


def test_exact_original_cuda_r2_format_rejection_is_bounded_and_retains_both_receipts(tmp_path):
    from hephaestus.storage.verified import read_verified_json
    service,request,attempted,store=_controller_rejected_format_fixture(tmp_path)
    result=service.execute(request)
    assert len(attempted)==2
    assert attempted[0].evidence==attempted[1].evidence
    assert attempted[0].run_id==attempted[1].run_id
    assert "FORMAT-ONLY RETRY:" in attempted[1].situation
    assert "primary_variable MUST" in attempted[1].situation
    assert "approval_status" in attempted[1].situation
    assert result.status=="completed"
    advisory=result.records[-1].payload
    assert advisory["bounded_format_repair_attempts"]==1
    assert len(advisory["rejected_inference_receipt_refs"])==1
    assert advisory["deterministic_action_allowed"] is False
    assert advisory["effective_action"] is None
    assert advisory["promotion_authorized"] is False
    rejected=read_verified_json(store,advisory["rejected_inference_receipt_refs"][0])
    assert rejected["error"]=="role response has invalid primary_variable"
    assert rejected["status"]=="rejected"
    accepted=read_verified_json(store,advisory["inference_receipt_ref"])
    assert accepted["status"]=="accepted"
    assert advisory["inference_receipt_ref"] != advisory["rejected_inference_receipt_refs"][0]


def test_repeated_invalid_primary_variable_fails_closed_after_exactly_one_format_repair(tmp_path):
    service,request,attempted,_=_controller_rejected_format_fixture(
        tmp_path,second_attempt="rejected")
    with pytest.raises(ValueError,match="invalid primary_variable"):
        service.execute(request)
    assert len(attempted)==2


def test_unrelated_value_error_does_not_retry_original_controller(tmp_path):
    service,request,attempted,_=_controller_rejected_format_fixture(tmp_path)
    class BadReasoner:
        def propose(self,*args,**kwargs):
            attempted.append(args[0])
            raise ValueError("role evidence is not independently verified")
    service.reasoner=BadReasoner()
    with pytest.raises(ValueError,match="independently verified"):
        service.execute(request)
    assert len(attempted)==1


def test_format_repair_without_verified_rejected_receipt_stays_denied(tmp_path):
    service,request,attempted,_=_controller_rejected_format_fixture(tmp_path)
    class ForgedFailure:
        def propose(self,*args,**kwargs):
            attempted.append(args[0])
            raise ValueError("role response has invalid primary_variable")
    service.reasoner=ForgedFailure()
    with pytest.raises(PermissionError,match="lacks immutable receipt"):
        service.execute(request)
    assert len(attempted)==1
