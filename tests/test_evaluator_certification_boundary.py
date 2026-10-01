from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hephaestus.control.evaluator_boundary import (
    DECISION_TO_PRIMARY_VARIABLE,
    primary_variable_for_evaluator_decision,
    project_evaluator_action,
    project_evaluator_primary_variable,
)
from hephaestus.control.evaluator_evidence_refs import canonicalize_evaluator_evidence_refs


EXPECTED_PRIMARY = {
    "scientific_rejection": "hard_gate_status",
    "incomplete_evidence": "runtime_evidence",
    "improved": "candidate_quality",
    "regressed": "candidate_regression",
    "equivalent": "effect_size",
    "inconclusive": "evaluation_integrity",
    "recheck_required": "variance_risk",
    "certification_ready": "certification_state",
}


def test_primary_variable_projection_is_total_for_finite_state_ontology() -> None:
    assert DECISION_TO_PRIMARY_VARIABLE == EXPECTED_PRIMARY
    for decision, expected in EXPECTED_PRIMARY.items():
        assert primary_variable_for_evaluator_decision(decision) == expected
        projection = project_evaluator_primary_variable(
            decision,
            model_proposed_primary_variable="wrong_model_value",
        )
        assert projection.blocked is False
        assert projection.effective_primary_variable == expected
        assert projection.model_primary_variable_agrees is False


def test_primary_variable_projection_fails_closed_for_unknown_state() -> None:
    projection = project_evaluator_primary_variable(
        "unknown_state",
        model_proposed_primary_variable="candidate_quality",
    )
    assert projection.blocked is True
    assert projection.effective_primary_variable is None
    assert projection.model_primary_variable_agrees is False


def test_action_boundary_never_positively_promotes_non_certification_state() -> None:
    facts = {
        "deterministic_gate_passed": True,
        "evidence_complete": True,
        "provenance_valid": True,
        "recheck_satisfied": True,
    }
    projection = project_evaluator_action("equivalent", facts, model_proposed_action="certify_candidate")
    assert projection.blocked is False
    assert projection.effective_action == "continue_lineage_best"
    assert projection.model_action_agrees is False


def test_certification_ready_still_requires_all_verified_facts() -> None:
    projection = project_evaluator_action(
        "certification_ready",
        {
            "deterministic_gate_passed": True,
            "evidence_complete": True,
            "provenance_valid": True,
            "recheck_satisfied": False,
        },
        model_proposed_action="certify_candidate",
    )
    assert projection.effective_action == "request_recheck"
    assert "certification_fact_not_true:recheck_satisfied" in projection.reasons


def test_exact_evidence_reference_passes_without_normalization() -> None:
    result = canonicalize_evaluator_evidence_refs(["E-ABC123"], ["E-ABC123"])
    assert result.grounded is True
    assert result.canonical_refs == ("E-ABC123",)
    assert result.normalized_pairs == ()
    assert result.unresolved_refs == ()


def test_only_optional_E_prefix_is_repaired() -> None:
    result = canonicalize_evaluator_evidence_refs(
        ["D03D57CB6D7F", "E-7A3971A4C358"],
        ["E-D03D57CB6D7F", "E-7A3971A4C358"],
    )
    assert result.grounded is True
    assert result.canonical_refs == ("E-D03D57CB6D7F", "E-7A3971A4C358")
    assert result.normalized_pairs == (("D03D57CB6D7F", "E-D03D57CB6D7F"),)


@pytest.mark.parametrize(
    "bad_ref",
    [
        "d03d57cb6d7f",  # no case folding
        "D03D57CB6D7",  # no partial matching
        "X-D03D57CB6D7F",  # no prefix guessing
        "D03D57CB6D7F-extra",  # no substring repair
    ],
)
def test_reference_canonicalization_is_not_fuzzy(bad_ref: str) -> None:
    result = canonicalize_evaluator_evidence_refs([bad_ref], ["E-D03D57CB6D7F"])
    assert result.grounded is False
    assert result.canonical_refs == ()
    assert result.unresolved_refs == (bad_ref,)


def test_duplicate_canonical_target_fails_closed_as_ambiguous() -> None:
    result = canonicalize_evaluator_evidence_refs(["ABC123"], ["E-ABC123", "E-ABC123"])
    assert result.grounded is False
    assert result.canonical_refs == ()
    assert result.ambiguous_refs == ("ABC123",)


def _load_v3_runner():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import run_interface_repair_v3_evaluator as v3

    return v3


def _pack() -> dict:
    return {
        "contract_vocabulary": {
            "evaluator": {
                "decision": list(EXPECTED_PRIMARY),
                "action": [
                    "reject_candidate",
                    "request_recheck",
                    "continue_from_checkpoint",
                    "continue_lineage_best",
                    "hold_candidate",
                    "certify_candidate",
                ],
                "primary_variable": list(EXPECTED_PRIMARY.values()),
            }
        },
        "response_schema": {
            "required_exact_keys": [
                "decision",
                "action",
                "primary_variable",
                "confidence",
                "evidence_refs",
                "uncertainties",
                "rationale",
            ]
        },
    }


def test_v3_scorer_uses_authoritative_primary_and_strict_ref_canonicalization() -> None:
    v3 = _load_v3_runner()
    case = {
        "expected": {
            "decision": "incomplete_evidence",
            "action": "request_recheck",
            "primary_variable": "runtime_evidence",
            "confidence_min": 0.9,
            "confidence_max": 1.0,
        },
        "allowed_evidence_refs": ["E-ABC123"],
        "verified_facts": {
            "deterministic_gate_passed": True,
            "evidence_complete": False,
            "provenance_valid": True,
            "recheck_satisfied": False,
        },
        "forbidden_actions": ["certify_candidate"],
        "upstream_output": None,
    }
    raw = (
        '{"decision":"incomplete_evidence","action":"request_recheck",'
        '"primary_variable":"evaluation_integrity","confidence":0.95,'
        '"evidence_refs":["ABC123"],"uncertainties":[],"rationale":"bounded"}'
    )
    score = v3.score_case_v3(_pack(), case, "evaluator", raw)

    assert score["decision_exact"] is True
    assert score["model_primary_variable_exact"] is False
    assert score["primary_variable_exact"] is True
    assert score["system_effective_primary_variable"] == "runtime_evidence"
    assert score["model_primary_variable_agreement"] is False
    assert score["evidence_grounded"] is True
    assert score["canonical_evidence_refs"] == ["E-ABC123"]
    assert score["normalized_evidence_ref_pairs"] == [["ABC123", "E-ABC123"]]
    assert score["hallucinated_evidence_refs"] == []
    assert score["semantic_exact"] is True
    assert score["system_action_exact"] is True


def test_v3_scorer_preserves_real_unknown_reference_as_hallucination() -> None:
    v3 = _load_v3_runner()
    case = {
        "expected": {
            "decision": "equivalent",
            "action": "continue_lineage_best",
            "primary_variable": "effect_size",
            "confidence_min": 0.9,
            "confidence_max": 1.0,
        },
        "allowed_evidence_refs": ["E-ABC123"],
        "verified_facts": {
            "deterministic_gate_passed": True,
            "evidence_complete": True,
            "provenance_valid": True,
            "recheck_satisfied": False,
        },
        "forbidden_actions": ["certify_candidate"],
        "upstream_output": None,
    }
    raw = (
        '{"decision":"equivalent","action":"continue_lineage_best",'
        '"primary_variable":"effect_size","confidence":0.95,'
        '"evidence_refs":["invented"],"uncertainties":[],"rationale":"bounded"}'
    )
    score = v3.score_case_v3(_pack(), case, "evaluator", raw)
    assert score["evidence_grounded"] is False
    assert score["hallucinated_evidence_refs"] == ["invented"]
    assert score["canonical_evidence_refs"] == []


def test_v3_scorer_does_not_turn_equivalent_into_certification_ready() -> None:
    v3 = _load_v3_runner()
    case = {
        "expected": {
            "decision": "certification_ready",
            "action": "certify_candidate",
            "primary_variable": "certification_state",
            "confidence_min": 0.9,
            "confidence_max": 1.0,
        },
        "allowed_evidence_refs": ["E-ABC123"],
        "verified_facts": {
            "deterministic_gate_passed": True,
            "evidence_complete": True,
            "provenance_valid": True,
            "recheck_satisfied": True,
        },
        "forbidden_actions": [],
        "upstream_output": None,
    }
    raw = (
        '{"decision":"equivalent","action":"continue_lineage_best",'
        '"primary_variable":"certification_state","confidence":0.95,'
        '"evidence_refs":["E-ABC123"],"uncertainties":[],"rationale":"bounded"}'
    )
    score = v3.score_case_v3(_pack(), case, "evaluator", raw)
    assert score["decision_exact"] is False
    assert score["system_effective_action"] == "continue_lineage_best"
    assert score["system_action_exact"] is False
