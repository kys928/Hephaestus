"""The original certified role and historical semantic-repair tasks must never mix."""
from __future__ import annotations

import json

import pytest

from hephaestus.evaluation.granite_dev_scoring import (
    GraniteScoringError, PRODUCTION_ONTOLOGY, RESEARCH_ONTOLOGY,
    compare_matched_development, score_development_decisions,
)


def _raw(decision: str, *, override: dict | None = None) -> str:
    val = {
        "decision": decision,
        "action": "retain_baseline",
        "primary_variable": "candidate_quality",
        "confidence": 0.5,
        "evidence_refs": ["dev:exposed"],
        "uncertainties": [],
        "rationale": "The supplied evidence is insufficient to assert an upgrade.",
    }
    val.update(override or {})
    return json.dumps(val, separators=(",", ":"))


def _ref(cid: str, ontology: str, label: str, *, partition="exposed_development") -> dict:
    return {
        "case_id": cid, "ontology": ontology, "partition": partition,
        "expected_decision": label,
    }


def _pred(cid: str, ontology: str, raw: str, *, partition="exposed_development") -> dict:
    return {
        "case_id": cid, "ontology": ontology, "partition": partition,
        "raw_generation": raw,
    }


def test_production_six_state_exact_decision():
    refs = [_ref("a", PRODUCTION_ONTOLOGY, "no_material_change"),
            _ref("b", PRODUCTION_ONTOLOGY, "evaluation_invalid")]
    pred = [_pred("a", PRODUCTION_ONTOLOGY, _raw("no_material_change")),
            _pred("b", PRODUCTION_ONTOLOGY, _raw("regressed"))]
    res = score_development_decisions(refs, pred, ontology=PRODUCTION_ONTOLOGY)
    assert res["count"] == 2 and res["decision_correct"] == 1
    assert res["decision_accuracy"] == 0.5
    assert res["per_state"]["evaluation_invalid"]["confusion"] == {"regressed": 1}
    assert res["can_promote"] is False
    assert res["model_execution_provenance_verified"] is False


def test_historical_eight_state_is_not_production_six_state():
    refs = [_ref("a", RESEARCH_ONTOLOGY, "equivalent"),
            _ref("b", RESEARCH_ONTOLOGY, "recheck_required"),
            _ref("c", RESEARCH_ONTOLOGY, "certification_ready")]
    pred = [_pred("a", RESEARCH_ONTOLOGY, _raw("equivalent")),
            _pred("b", RESEARCH_ONTOLOGY, _raw("recheck_required")),
            _pred("c", RESEARCH_ONTOLOGY, _raw("certification_ready"))]
    out = score_development_decisions(refs, pred, ontology=RESEARCH_ONTOLOGY)
    assert out["decision_correct"] == 3
    assert out["per_state"]["equivalent"]["correct"] == 1
    with pytest.raises(GraniteScoringError, match="cross-ontology"):
        score_development_decisions(refs, pred, ontology=PRODUCTION_ONTOLOGY)
    with pytest.raises(GraniteScoringError, match="different ontology"):
        compare_matched_development(refs, pred, pred,
            reference_ontology=RESEARCH_ONTOLOGY,
            parent_ontology=RESEARCH_ONTOLOGY,
            candidate_ontology=PRODUCTION_ONTOLOGY)


@pytest.mark.parametrize("raw,reason", [
    ('{"decision":"equivalent","decision":"recheck_required"}', "invalid_strict_json"),
    (_raw("equivalent").replace('"confidence":0.5', '"confidence":1e400'), "invalid_strict_json"),
    ('[{"decision":"equivalent"}]', "invalid_envelope"),
    ("\x60"*3 + "json\n" + _raw("equivalent") + "\n" + "\x60"*3, "invalid_strict_json"),
    (_raw("equivalent", override={"confidence": True}), "invalid_envelope"),
    (_raw("equivalent", override={"extra": 1}), "invalid_envelope"),
    (_raw("equivalent", override={"uncertainties": "none"}), "invalid_envelope"),
    (_raw("no_material_change"), "decision_outside_ontology"),
])
def test_wrong_format_never_receives_credit(raw, reason):
    res = score_development_decisions(
        [_ref("a", RESEARCH_ONTOLOGY, "equivalent")],
        [_pred("a", RESEARCH_ONTOLOGY, raw)],
        ontology=RESEARCH_ONTOLOGY,
    )
    assert res["decision_correct"] == 0
    assert res["invalid_reasons"] == {reason: 1}


def test_matched_dev_result_is_explicitly_nonconfirmatory():
    refs = [_ref("a", RESEARCH_ONTOLOGY, "equivalent"),
            _ref("b", RESEARCH_ONTOLOGY, "recheck_required")]
    parent = [_pred("a", RESEARCH_ONTOLOGY, _raw("regressed")),
              _pred("b", RESEARCH_ONTOLOGY, _raw("recheck_required"))]
    candidate = [_pred("a", RESEARCH_ONTOLOGY, _raw("equivalent")),
                 _pred("b", RESEARCH_ONTOLOGY, _raw("recheck_required"))]
    res = compare_matched_development(
        refs, parent, candidate,
        reference_ontology=RESEARCH_ONTOLOGY,
        parent_ontology=RESEARCH_ONTOLOGY,
        candidate_ontology=RESEARCH_ONTOLOGY,
    )
    assert res["absolute_decision_accuracy_gain"] == 0.5
    assert res["promotion_eligible"] is False
    assert res["independent_confirmation"] is False


@pytest.mark.parametrize("action", [
    "missing_prediction", "extra_prediction", "duplicate_case",
    "wrong_partition", "invalid_reference_state", "disallowed_partition",
])
def test_incomplete_or_protected_inputs_fail_closed(action):
    refs = [_ref("a", RESEARCH_ONTOLOGY, "equivalent")]
    pred = [_pred("a", RESEARCH_ONTOLOGY, _raw("equivalent"))]
    if action == "missing_prediction":
        refs.append(_ref("b", RESEARCH_ONTOLOGY, "regressed"))
    elif action == "extra_prediction":
        pred.append(_pred("b", RESEARCH_ONTOLOGY, _raw("regressed")))
    elif action == "duplicate_case":
        pred.append(dict(pred[0]))
    elif action == "wrong_partition":
        pred[0]["partition"] = "burned_diagnostic"
    elif action == "invalid_reference_state":
        refs[0]["expected_decision"] = "no_material_change"
    else:
        refs[0]["partition"] = "protected_holdout"
        pred[0]["partition"] = "protected_holdout"
    with pytest.raises(GraniteScoringError):
        score_development_decisions(refs, pred, ontology=RESEARCH_ONTOLOGY)


def test_no_reflection_of_protected_predictions_in_aggregate():
    refs = [_ref("a", RESEARCH_ONTOLOGY, "equivalent")]
    raw = _raw("equivalent", override={"rationale": "private scenario text"})
    result = score_development_decisions(
        refs, [_pred("a", RESEARCH_ONTOLOGY, raw)], ontology=RESEARCH_ONTOLOGY,
    )
    assert "private scenario text" not in json.dumps(result)
    assert result["historical_exposed_only"] is True
