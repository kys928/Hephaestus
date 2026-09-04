from __future__ import annotations

from hephaestus.planning import ResolvedEvidenceExperimentPlanner
from hephaestus.schemas.diagnosis_contract import DiagnosisReport, DiagnosticHypothesis


def _diagnosis(*, status: str, resolved: bool) -> DiagnosisReport:
    hypothesis = DiagnosticHypothesis(
        hypothesis_id="hyp-data-coverage",
        failure_domain="data_coverage",
        summary="Explicit coverage evidence indicates a possible data-coverage gap.",
        supporting_evidence_refs=["evidence://dataset-coverage"],
        required_tests=["measure capability coverage against manifest domains"],
        recommended_intervention_kinds=["replace_or_mix_dataset", "collect_more_evidence"],
        confidence=0.825,
    )
    return DiagnosisReport(
        report_id="diag-targeted",
        request_id="request-targeted",
        run_id="run-1",
        lineage_id="lineage-1",
        stage_name="smoke_test",
        status=status,
        hypotheses=[hypothesis],
        leading_hypothesis_id=hypothesis.hypothesis_id,
        confidence=0.825,
        metadata={
            "baseline_ref": "run://baseline",
            "baseline_quality": 0.95,
            "resolved_intervention_kinds": ["collect_more_evidence"] if resolved else [],
        },
    )


def test_completed_targeted_diagnostics_do_not_immediately_repeat_collect_more_evidence() -> None:
    proposals = list(
        ResolvedEvidenceExperimentPlanner().propose_interventions(
            _diagnosis(status="completed", resolved=True)
        )
    )

    assert proposals
    assert proposals[0].intervention_kind == "replace_or_mix_dataset"
    assert proposals[0].primary_variable == "dataset_mixture"
    assert all(item.intervention_kind != "collect_more_evidence" for item in proposals)
    assert proposals[0].metadata["resolution_boundary_applied"] is True
    assert proposals[0].metadata["resolved_intervention_kinds_skipped"] == [
        "collect_more_evidence"
    ]


def test_inconclusive_diagnosis_keeps_collect_more_evidence_available() -> None:
    proposals = list(
        ResolvedEvidenceExperimentPlanner().propose_interventions(
            _diagnosis(status="inconclusive", resolved=True)
        )
    )

    assert any(item.intervention_kind == "collect_more_evidence" for item in proposals)
