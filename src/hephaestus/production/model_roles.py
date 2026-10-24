"""Explicit opt-in model proposals inside the existing staged services.

Never treats language-model output as a verified fact or an applied action.
A delegate performs the deterministic service work; selected stages then get
an additional *real*, evidence-validated role-model proposal and audit record.
Model failure blocks that stage rather than silently skipping the model.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from hephaestus.control.staged_state import (
    StagedAutonomousServices, StagedOperationRequest, StagedOperationResult,
    StagedOutputRecord, StagedOperationService,
)
from hephaestus.control.deterministic_role_boundary import (
    DeterministicRoleContext, EvidenceRegistry,
)
from hephaestus.llm.role_contract import RoleTask
from hephaestus.llm.certified_runtime import CertifiedRoleReasoner
from hephaestus.policy.action_registry import canonical_action_name


ROLE_SUBSTEP = {
    "controller": "judge_entry_decision",
    "diagnosis": "evidence_based_diagnosis",
    "planner": "intervention_ranking",
    "evaluator": "semantic_comparison",
    "judge": "governed_verdict",
}


class ModelEvidenceBinder(Protocol):
    def bind(self, *, role: str, request: StagedOperationRequest
             ) -> tuple[RoleTask, EvidenceRegistry, DeterministicRoleContext]: ...


@dataclass(slots=True)
class ModelBackedStage:
    role: str
    delegate: StagedOperationService
    reasoner: CertifiedRoleReasoner
    evidence: ModelEvidenceBinder

    def execute(self, request: StagedOperationRequest) -> StagedOperationResult:
        if request.substep != ROLE_SUBSTEP[self.role]:
            return self.delegate.execute(request)
        # The deterministic service must produce a valid result first.
        # The model can review or challenge it, never mint its validity.
        base = self.delegate.execute(request)
        if base.status != "completed":
            return base
        task, registry, context = self.evidence.bind(role=self.role, request=request)
        if task.role != self.role:
            raise ValueError("evidence binder supplied a different role")
        observation = self.reasoner.propose(task, registry, context=context)
        gate = observation.action_gate
        if gate is not None and not gate.allowed:
            return replace(base, status="blocked", resumable=False,
                           blocking_issues=(*base.blocking_issues, "role_model_action_gate_blocked",
                                            *gate.failures))
        if self.role == "judge":
            # Both a human-auditable model opinion and a preexisting governed
            # Judge verdict are required, with no silent override of either.
            selected = canonical_action_name(str(base.metadata.get("proposed_action") or ""))
            proposed = canonical_action_name(observation.answer.action)
            if not selected or selected != proposed:
                return replace(base, status="blocked", resumable=False,
                               blocking_issues=(*base.blocking_issues,
                                                "model_judge_disagreed_with_governed_verdict"))
        record = StagedOutputRecord(
            "model_role_proposal", {
                "role": observation.role,
                "decision": observation.answer.decision,
                "action": observation.answer.action,
                "primary_variable": observation.answer.primary_variable,
                "confidence": observation.answer.confidence,
                "evidence_refs": list(observation.answer.evidence_refs),
                "uncertainties": list(observation.answer.uncertainties),
                "rationale": observation.answer.rationale,
                "model_id": observation.model_id,
                "base_revision": observation.revision,
                "adapter_sha256": observation.adapter_sha256,
                "prompt_sha256": observation.prompt_sha256,
                "output_sha256": observation.output_sha256,
                "inference_receipt_ref": observation.inference_receipt_ref,
                "advisory_only": True,
            }
        )
        return replace(base, records=(*base.records, record),
                       output_refs=(*base.output_refs, observation.inference_receipt_ref),
                       metadata={**base.metadata,
                                 "model_role": observation.role,
                                 "model_role_output_sha256": observation.output_sha256,
                                 "inference_receipt_ref": observation.inference_receipt_ref,
                                 "model_role_proposal_advisory_only": True})


def with_model_backed_roles(
    services: StagedAutonomousServices, reasoner: CertifiedRoleReasoner,
    evidence: ModelEvidenceBinder, *, enabled: bool = False,
) -> StagedAutonomousServices:
    """Explicitly turn on role proposals; leave unconfigured services blocked."""
    if not enabled:
        raise PermissionError("model-backed staged execution requires explicit enable=True")
    mapping = {
        "controller": "judge_entry",
        "diagnosis": "diagnosis",
        "planner": "planner",
        "evaluator": "evaluator",
        "judge": "judge_exit",
    }
    kwargs = {}
    for role, field in mapping.items():
        service = getattr(services, field)
        if service is None:
            raise ValueError(f"cannot wire missing {role} service")
        kwargs[field] = ModelBackedStage(role, service, reasoner, evidence)
    return replace(services, **kwargs)
