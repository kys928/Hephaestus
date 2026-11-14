"""CPU-only calibration regressions; not a claim of successful CUDA reasoning."""
from copy import deepcopy

import pytest

from hephaestus.development.prometheus_scientific_decision import (
    _independent_evidence_calibration, prepare,
)
from hephaestus.llm.role_contract import RoleAnswer
from hephaestus.roles.diagnostician import select_model_hypothesis
from hephaestus.schemas.diagnosis_contract import DiagnosticHypothesis
from test_prometheus_scientific_decision import _reports


def test_historical_evidence_calibration_does_not_invent_eval_failure():
    report, _, _ = prepare(*_reports())
    guidance = _independent_evidence_calibration(report)
    assert report.status == "inconclusive"
    assert "positively supported non-inconclusive failure domains=[]" in guidance
    assert "eval_integrity_unverified" in guidance
    assert "NOT an observation that the evaluation protocol failed" in guidance
    assert "historical" in guidance
    assert "independent admission gate" in guidance


def test_calibration_changes_with_real_independent_support_not_fixed_answer():
    report, _, _ = prepare(*_reports())
    baseline = _independent_evidence_calibration(report)
    changed = deepcopy(report)
    changed.hypotheses.append(DiagnosticHypothesis(
        hypothesis_id="controlled-support",
        failure_domain="data_coverage",
        summary="Synthetic test fixture only",
        supporting_evidence_refs=["sha256:" + "a" * 64],
        contradicting_evidence_refs=[],
        confidence=0.8,
    ))
    assert "failure domains=[\"data_coverage\"]" in _independent_evidence_calibration(changed)
    changed.hypotheses[-1].contradicting_evidence_refs = ["sha256:" + "b" * 64]
    assert _independent_evidence_calibration(changed) == baseline


def test_calibration_never_weakens_independent_hypothesis_admission():
    report, _, _ = prepare(*_reports())
    unsupported = RoleAnswer(
        "evaluation_integrity", "repair_evaluation", "evaluation_protocol",
        0.81, ("sha256:" + "a" * 64,), ("missing scorecard",),
        "Evaluation integrity is defective.",
    )
    with pytest.raises(ValueError, match="unsupported"):
        select_model_hypothesis(report, unsupported, "sha256:" + "b" * 64)
    admissible = RoleAnswer(
        "inconclusive", "collect_more_evidence", "evidence",
        0.45, ("sha256:" + "a" * 64,), ("not independently verified",),
        "Existing observations do not establish an original parent failure.",
    )
    assert select_model_hypothesis(report, admissible, "sha256:" + "b" * 64).status == "inconclusive"
