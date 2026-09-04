"""Planner wrapper that advances past interventions already resolved by new evidence."""
from __future__ import annotations

from typing import Sequence

from hephaestus.schemas.diagnosis_contract import DiagnosisReport
from hephaestus.schemas.experiment_contract import InterventionProposal

from .service import ClosedLoopExperimentPlanner


class ResolvedEvidenceExperimentPlanner(ClosedLoopExperimentPlanner):
    """Closed-loop planner with an explicit same-cycle resolution boundary.

    A diagnosis producer may record ``metadata['resolved_intervention_kinds']``
    after executing a diagnostic-only intervention such as
    ``collect_more_evidence``. When the diagnosis is now completed with no
    missing or blocking evidence, the planner does not immediately select that
    already-fulfilled intervention again. All unresolved interventions retain
    the base planner's scoring and ordering.
    """

    def propose_interventions(
        self, diagnosis: DiagnosisReport
    ) -> Sequence[InterventionProposal]:
        proposals = list(super().propose_interventions(diagnosis))
        resolved = {
            str(item)
            for item in diagnosis.metadata.get("resolved_intervention_kinds", [])
            if str(item)
        }
        ready = (
            diagnosis.status == "completed"
            and not diagnosis.missing_evidence
            and not any(issue.blocking for issue in diagnosis.issues)
        )
        if not ready or not resolved:
            return proposals

        remaining = [
            proposal
            for proposal in proposals
            if proposal.intervention_kind not in resolved
        ]
        if not remaining:
            return proposals

        for rank, proposal in enumerate(remaining, start=1):
            proposal.metadata["rank"] = rank
            proposal.metadata["resolved_intervention_kinds_skipped"] = sorted(resolved)
            proposal.metadata["resolution_boundary_applied"] = True
        return remaining


__all__ = ["ResolvedEvidenceExperimentPlanner"]
