from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from hephaestus.schemas.contract_common import ContractIssue
from hephaestus.schemas.discovery_contract import (
    DatasetCandidate,
    DatasetSearchRequest,
    DatasetSelectionDecision,
)
from hephaestus.utils.hashing import hash_json

from .audit import CandidateAudit, audit_candidate
from .normalization import normalize_dataset_candidate


@dataclass(slots=True)
class DeterministicDatasetSelectionService:
    """Evidence-preserving deterministic candidate selector."""

    minimum_score: float = 0.45

    def select(
        self,
        request: DatasetSearchRequest,
        candidates: Sequence[DatasetCandidate],
    ) -> DatasetSelectionDecision:
        normalized = [normalize_dataset_candidate(candidate) for candidate in candidates]
        audits: dict[str, CandidateAudit] = {
            candidate.candidate_id: audit_candidate(request, candidate) for candidate in normalized
        }
        ranked = sorted(normalized, key=lambda item: (-audits[item.candidate_id].score, item.candidate_id))
        ranked_ids = [candidate.candidate_id for candidate in ranked]

        rejected: dict[str, str] = {}
        eligible: list[DatasetCandidate] = []
        approval_blocked: list[DatasetCandidate] = []
        for candidate in ranked:
            audit = audits[candidate.candidate_id]
            candidate.score_components = dict(audit.score_components)
            if audit.rejected_reasons:
                rejected[candidate.candidate_id] = ";".join(audit.rejected_reasons)
            elif audit.score < self.minimum_score:
                rejected[candidate.candidate_id] = f"score_below_threshold:{audit.score:.8f}<{self.minimum_score:.8f}"
            elif audit.required_approvals:
                approval_blocked.append(candidate)
                rejected[candidate.candidate_id] = "approval_required:" + ",".join(audit.required_approvals)
            else:
                eligible.append(candidate)

        selected: list[DatasetCandidate] = []
        max_selected = max(1, int(request.metadata.get("max_selected_candidates", 1) or 1))
        mixture_delta = max(0.0, float(request.metadata.get("mixture_score_delta", 0.05) or 0.05))
        if eligible:
            best_score = audits[eligible[0].candidate_id].score
            selected = [
                candidate
                for candidate in eligible
                if best_score - audits[candidate.candidate_id].score <= mixture_delta
            ][:max_selected]

        issues: list[ContractIssue] = []
        if selected:
            status = "selected"
        elif approval_blocked:
            status = "blocked"
            issues.append(
                ContractIssue(
                    code="dataset_selection_approval_required",
                    category="approval_required",
                    message="acceptable candidates require explicit approval before selection",
                    blocking=True,
                    evidence_refs=list(request.evidence_refs),
                )
            )
        else:
            status = "inconclusive"
            issues.append(
                ContractIssue(
                    code="no_acceptable_dataset_candidate",
                    category="candidate_not_found",
                    message="no candidate satisfied compatibility, policy, and score requirements",
                    retryable=True,
                    blocking=True,
                    evidence_refs=list(request.evidence_refs),
                )
            )
        decision_approvals = sorted(
            {
                approval
                for candidate in approval_blocked
                for approval in audits[candidate.candidate_id].required_approvals
            }
        ) if status == "blocked" else []

        selected_ids = [candidate.candidate_id for candidate in selected]
        positive_scores = {item.candidate_id: max(audits[item.candidate_id].score, 1e-12) for item in selected}
        total = sum(positive_scores.values())
        mixture_weights = {
            candidate_id: round(score / total, 12) for candidate_id, score in positive_scores.items()
        }
        if mixture_weights:
            last_id = selected_ids[-1]
            mixture_weights[last_id] = round(1.0 - sum(mixture_weights[item] for item in selected_ids[:-1]), 12)

        preprocessing = sorted(
            {
                requirement
                for candidate in selected
                for requirement in audits[candidate.candidate_id].preprocessing_requirements
            }
        )
        top_score = audits[ranked_ids[0]].score if ranked_ids else 0.0
        second_score = audits[ranked_ids[1]].score if len(ranked_ids) > 1 else 0.0
        confidence = 0.0 if not selected else min(1.0, top_score * (0.8 + 0.2 * max(0.0, top_score - second_score)))
        evidence_refs = sorted(
            {
                *request.evidence_refs,
                *(ref for candidate in selected for ref in candidate.evidence_refs),
            }
        )
        decision_seed = {
            "request": request.to_dict(),
            "ranked_candidate_ids": ranked_ids,
            "audits": {candidate_id: audits[candidate_id].to_dict() for candidate_id in sorted(audits)},
            "status": status,
            "selected_candidate_ids": selected_ids,
        }
        return DatasetSelectionDecision(
            decision_id=f"dataset-selection-{hash_json(decision_seed)[:20]}",
            request_id=request.request_id,
            status=status,
            selected_candidate_ids=selected_ids,
            ranked_candidate_ids=ranked_ids,
            rejected_candidates=rejected,
            selection_rationale=(
                "selected highest-scoring policy-compatible candidate(s)"
                if selected
                else "selection blocked pending explicit approval"
                if approval_blocked
                else "evidence did not support an acceptable candidate"
            ),
            mixture_weights=mixture_weights,
            preprocessing_requirements={"operations": preprocessing},
            required_approvals=decision_approvals,
            evidence_refs=evidence_refs,
            issues=issues,
            confidence=round(confidence, 8),
            metadata={
                "selection_algorithm": "deterministic-weighted-v1",
                "minimum_score": self.minimum_score,
                "material_candidates": [candidate.to_dict() for candidate in ranked],
                "candidate_audits": {
                    candidate_id: audits[candidate_id].to_dict() for candidate_id in sorted(audits)
                },
            },
        )
