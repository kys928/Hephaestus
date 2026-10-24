"""Explicit model proposal/admission, followed by governed staged execution."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from hephaestus.control.staged_state import (
    StagedAutonomousServices, StagedOperationRequest, StagedOperationResult,
    StagedOutputRecord, StagedOperationService,
)
from hephaestus.control.deterministic_role_boundary import DeterministicRoleContext, EvidenceRegistry
from hephaestus.llm.role_contract import RoleTask
from hephaestus.llm.certified_runtime import CertifiedRoleReasoner
from hephaestus.llm.inference_receipts import verify_inference_receipt
from hephaestus.planning.service import ClosedLoopExperimentPlanner
from hephaestus.policy.action_registry import canonical_action_name
from hephaestus.policy.model_proposals import planner_selection
from hephaestus.roles.diagnostician import select_model_hypothesis
from hephaestus.schemas.diagnosis_contract import DiagnosisReport
from hephaestus.schemas.experiment_contract import InterventionProposal
from hephaestus.schemas.model_proposal import ModelInterventionSelection
from hephaestus.storage.verified import canonical_bytes, read_verified_json

ROLE_SUBSTEP = {"controller":"judge_entry_decision", "diagnosis":"evidence_based_diagnosis",
    "planner":"intervention_ranking", "evaluator":"semantic_comparison", "judge":"governed_verdict"}


class ModelEvidenceBinder(Protocol):
    def bind(self, *, role: str, request: StagedOperationRequest
             ) -> tuple[RoleTask, EvidenceRegistry, DeterministicRoleContext]: ...

    def diagnosis_for(self, request: StagedOperationRequest) -> DiagnosisReport: ...

    def verify_refs(self, request: StagedOperationRequest, refs: list[str]) -> set[str]: ...


@dataclass(slots=True)
class ModelBackedStage:
    role: str
    delegate: StagedOperationService
    reasoner: CertifiedRoleReasoner
    evidence: ModelEvidenceBinder
    planner: ClosedLoopExperimentPlanner

    def _selected(self, request):
        prior = request.prior_outputs.get("planner.intervention_ranking", {})
        doc = read_verified_json(self.reasoner.artifact_store, str(prior.get("selected_intervention_ref") or ""))
        receipt = verify_inference_receipt(self.reasoner.artifact_store, doc["selection"]["inference_receipt_ref"],
                                           self.reasoner.contract)
        if (receipt["role"] != "planner" or receipt["run_id"] != request.run_id or
                receipt["lineage_id"] != request.lineage_id or receipt["answer"] != doc["selection"]["model_decision"]):
            raise ValueError("selected intervention model receipt identity mismatch")
        selection = ModelInterventionSelection.from_dict(doc["selection"])
        answer = self.reasoner.contract.parse("planner", receipt["raw_output"], set(receipt["evidence_refs"]))
        expected = planner_selection(answer, selection.inference_receipt_ref)
        if expected.to_dict() != selection.to_dict():
            raise ValueError("selected intervention does not derive from the raw model decision")
        verified_refs = self.evidence.verify_refs(request, selection.evidence_refs)
        diagnosis = self.evidence.diagnosis_for(request)
        chosen = self.planner.admit_model_selection(diagnosis, selection, verified_refs)
        if chosen.to_dict() != doc["intervention"]:
            raise ValueError("selected intervention changed since deterministic admission")
        return diagnosis, chosen

    def _planned_step(self, request):
        diagnosis, chosen = self._selected(request)
        if request.substep == "experiment_proposal":
            proposal = self.planner.propose_experiment(diagnosis, chosen, None, None,
                                                       allow_unresolved_discovery=True)
            return StagedOperationResult(records=(StagedOutputRecord("experiment_proposal", proposal.to_dict()),),
                output_refs=(proposal.experiment_id,), metadata={"experiment_id":proposal.experiment_id,
                "intervention_id":chosen.intervention_id, "primary_variable":chosen.primary_variable,
                "unresolved_discovery":proposal.metadata["unresolved_discovery"]})
        if request.substep == "discovery_request_preparation":
            dataset, model = self.planner.create_discovery_requests(diagnosis, chosen)
            return StagedOperationResult(records=(StagedOutputRecord("search_request", {
                "dataset_request":dataset.to_dict() if dataset else None,
                "model_request":model.to_dict() if model else None,
                "discovery_required":bool(dataset or model)}),),
                metadata={"dataset_request_id":dataset.request_id if dataset else None,
                          "model_request_id":model.request_id if model else None})
        requirements = list(chosen.metadata.get("required_approvals", []))
        return StagedOperationResult(records=(StagedOutputRecord("approval_requirements", {
            "intervention_id":chosen.intervention_id, "required_approvals":requirements}),),
            metadata={"required_approvals":requirements})

    def execute(self, request: StagedOperationRequest) -> StagedOperationResult:
        if self.role == "planner" and request.substep in {
                "experiment_proposal", "discovery_request_preparation", "approval_requirement_discovery"}:
            return self._planned_step(request)
        if request.substep != ROLE_SUBSTEP[self.role]:
            return self.delegate.execute(request)
        # Controller/Planner must be admitted before any delegated operation.
        # Diagnosis, evaluation and Judge interpret deterministic observations;
        # these delegates must be pure observers. Mutation remains action-owner IO.
        before = self.role in {"controller", "planner"}
        base = StagedOperationResult() if before else self.delegate.execute(request)
        if base.status != "completed":
            return base
        task, registry, context = self.evidence.bind(role=self.role, request=request)
        if task.role != self.role:
            raise ValueError("evidence binder supplied a different role")
        observation = self.reasoner.propose(task, registry, context=context)
        gate = observation.action_gate
        if gate is not None and not gate.allowed:
            return StagedOperationResult(status="blocked", resumable=False,
                blocking_issues=("role_model_action_gate_blocked", *gate.failures))
        if self.role == "controller":
            if observation.answer.decision in {"blocked", "noop"} or observation.answer.action == "noop":
                return StagedOperationResult(status="blocked", resumable=False,
                    output_refs=(observation.inference_receipt_ref,), blocking_issues=("model_controller_refused_execution",))
            base = self.delegate.execute(request)
        elif self.role == "planner":
            diagnosis = self.evidence.diagnosis_for(request)
            selection = planner_selection(observation.answer, observation.inference_receipt_ref)
            chosen = self.planner.admit_model_selection(diagnosis, selection,
                set(registry.validate(selection.evidence_refs, run_id=request.run_id,
                                      lineage_id=request.lineage_id).valid_refs))
            if chosen.intervention_kind == "stop":
                return StagedOperationResult(status="cancelled", records=(StagedOutputRecord(
                    "intervention_proposal", chosen.to_dict()),), blocking_issues=("model_selected_stop",), resumable=False)
            artifact = self.reasoner.artifact_store.put_bytes(canonical_bytes({
                "intervention":chosen.to_dict(), "selection":selection.to_dict()}), media_type="application/json")
            base = StagedOperationResult(records=(StagedOutputRecord("intervention_proposal", chosen.to_dict()),),
                output_refs=(artifact.artifact_ref,), metadata={"selected_intervention_ref":artifact.artifact_ref,
                    "intervention_id":chosen.intervention_id, "primary_variable":chosen.primary_variable,
                    "required_approvals":chosen.metadata["required_approvals"]})
        elif self.role == "diagnosis":
            records = list(base.records)
            index = next((i for i,r in enumerate(records) if r.kind == "diagnosis_report"), None)
            if index is None:
                raise ValueError("model diagnosis requires an actual diagnosis report")
            report = DiagnosisReport.from_dict(records[index].payload)
            report = select_model_hypothesis(report, observation.answer, observation.inference_receipt_ref)
            records[index] = StagedOutputRecord("diagnosis_report", report.to_dict())
            base = replace(base, records=tuple(records), metadata={**base.metadata,
                "diagnosis_report_id":report.report_id, "missing_evidence":report.missing_evidence,
                "diagnosis_status":report.status, "leading_hypothesis_id":report.leading_hypothesis_id})
        elif self.role == "judge":
            selected = canonical_action_name(str(base.metadata.get("proposed_action") or ""))
            if not selected or selected != canonical_action_name(observation.answer.action):
                return replace(base, status="blocked", resumable=False,
                    blocking_issues=(*base.blocking_issues, "model_judge_disagreed_with_governed_verdict"))
        advisory = self.role not in {"planner", "diagnosis"}
        record = StagedOutputRecord("model_role_proposal", {
            "role":observation.role, **observation.answer.to_dict(), "model_id":observation.model_id,
            "base_revision":observation.revision, "adapter_sha256":observation.adapter_sha256,
            "prompt_sha256":observation.prompt_sha256, "output_sha256":observation.output_sha256,
            "inference_receipt_ref":observation.inference_receipt_ref, "advisory_only":advisory})
        return replace(base, records=(*base.records, record),
            output_refs=(*base.output_refs, observation.inference_receipt_ref),
            metadata={**base.metadata, "model_role":observation.role,
                "model_role_output_sha256":observation.output_sha256,
                "inference_receipt_ref":observation.inference_receipt_ref,
                "model_role_proposal_advisory_only":advisory})


def with_model_backed_roles(services: StagedAutonomousServices, reasoner: CertifiedRoleReasoner,
    evidence: ModelEvidenceBinder, *, enabled: bool = False,
    planner: ClosedLoopExperimentPlanner | None = None) -> StagedAutonomousServices:
    if not enabled:
        raise PermissionError("model-backed staged execution requires explicit enable=True")
    if planner is None:
        raise ValueError("model-backed execution requires the configured scientific Planner")
    mapping = {"controller":"judge_entry", "diagnosis":"diagnosis", "planner":"planner",
               "evaluator":"evaluator", "judge":"judge_exit"}
    kwargs = {}
    for role, field in mapping.items():
        service = getattr(services, field)
        if service is None:
            raise ValueError(f"cannot wire missing {role} service")
        kwargs[field] = ModelBackedStage(role, service, reasoner, evidence, planner)
    return replace(services, **kwargs)
