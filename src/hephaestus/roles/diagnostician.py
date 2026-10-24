"""Control-spine adapter for evidence-based diagnosis."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from hephaestus.policy.model_proposals import DIAGNOSIS_DOMAINS

from hephaestus.interfaces.services import DiagnosisService
from hephaestus.schemas.diagnosis_contract import DiagnosisReport, DiagnosisRequest


@dataclass(slots=True)
class DiagnosticianRole:
    service: DiagnosisService
    name: str = "diagnostician"

    def run(self, request: DiagnosisRequest) -> DiagnosisReport:
        """Recommend diagnostic findings without executing an intervention."""

        return self.service.diagnose(request)


def select_model_hypothesis(report: DiagnosisReport, answer, receipt_ref: str) -> DiagnosisReport:
    result = deepcopy(report)
    result.metadata["model_diagnosis"] = {"interpretation":answer.to_dict(),
        "inference_receipt_ref":receipt_ref, "machine_facts_changed":False}
    if answer.decision in {"inconclusive", "observability_limit"}:
        result.status = "inconclusive"
        result.confidence = min(result.confidence, answer.confidence)
        return result
    domains = DIAGNOSIS_DOMAINS[answer.decision]
    supported = [h for h in result.hypotheses if h.failure_domain in domains
                 and h.supporting_evidence_refs and not h.contradicting_evidence_refs]
    if not supported:
        raise ValueError("model diagnosis is unsupported by independently observed evidence")
    chosen = max(supported, key=lambda h:h.confidence)
    result.leading_hypothesis_id = chosen.hypothesis_id
    result.confidence = min(result.confidence, chosen.confidence, answer.confidence)
    # Missing evidence and blocking issues are deliberately preserved.
    return result
