"""Offline oracle attribution tests; no GPU, downloads, or S3 calls."""
from dataclasses import replace

import pytest
import build_evaluator_semantic_invariance_repair_v1 as frozen
from hephaestus.schemas.evaluator_factors import ScientificFactors, combination_space
from hephaestus.policy.evaluator_semantic_oracle import scientific_state
from evaluator_semantic_factor_attribution import (
    FIELDS, reference_factors, nearest_counterfactuals, analyze_observed,
    paired_comparison, score_explicit_factor_probe,
)


@pytest.fixture(scope="module")
def dev():
    return frozen.build_pack()["partitions"]["evaluator"]["development_diagnostic"]


def observed(case, prediction=None):
    expected=case["expected"]["decision"]
    got=expected if prediction is None else prediction
    return {"case_id":case["case_id"],"score":{
        "parsed":{"decision":got},"decision_exact":got==expected}}


@pytest.mark.parametrize("state", frozen.STATE_ORDER)
def test_each_reference_state_is_consistent_with_immutable_oracle_and_machine(dev,state):
    case=next(r for r in dev if r["semantic_root"]==state)
    actual=reference_factors(case)
    assert scientific_state(actual)==state
    assert len(FIELDS)==6
    assert len(combination_space())==96


def test_single_factor_neutral_to_inconclusive_has_provenance_counterfactual(dev):
    f=reference_factors(next(r for r in dev if r["semantic_root"]=="equivalent"))
    x=nearest_counterfactuals(f,"inconclusive")
    assert x["minimum_factor_edits"]==1
    assert x["necessary_factors"]==["provenance_valid"]


def test_single_factor_neutral_to_recheck_has_variance_counterfactual(dev):
    f=reference_factors(next(r for r in dev if r["semantic_root"]=="equivalent"))
    x=nearest_counterfactuals(f,"recheck_required")
    assert x["minimum_factor_edits"]==1
    assert x["necessary_factors"]==["variance_stable"]


def test_unstable_to_inconclusive_provenance_counterfactual(dev):
    f=reference_factors(next(r for r in dev if r["semantic_root"]=="recheck_required"))
    x=nearest_counterfactuals(f,"inconclusive")
    assert x["minimum_factor_edits"]==1
    assert x["necessary_factors"]==["provenance_valid"]


def test_nonstate_output_has_no_spurious_factor_inference(dev):
    f=reference_factors(dev[0])
    x=nearest_counterfactuals(f,"certify_candidate")
    assert x["minimum_factor_edits"] is None
    assert not x["known_state"]


def test_partial_observational_report_does_not_claim_extraction(dev):
    cases=[r for r in dev if r["semantic_root"]=="equivalent"][:3]
    rows=[observed(c,"inconclusive") for c in cases]
    audit=analyze_observed(dev,rows,"test")
    assert audit["correct"]==0
    assert audit["observed_cases"]==3
    assert audit["factors_necessary_in_every_minimal_explanation"]["provenance_valid"]==3
    assert audit["actual_factor_extraction_error_observed"] is False
    assert audit["actual_precedence_error_observed"] is False
    assert audit["by_reference_factor_value"]["effect"]["neutral"]["total"]==3


def test_missing_and_duplicate_samples_rejected(dev):
    with pytest.raises(ValueError, match="missing/duplicate/unknown"):
        analyze_observed(dev,[],"empty")
    with pytest.raises(ValueError, match="missing/duplicate/unknown"):
        analyze_observed(dev,[observed(dev[0]),observed(dev[0])],"duplicate")


def test_inconsistent_stored_score_rejected(dev):
    r=observed(dev[0]);r["score"]["decision_exact"]=False
    with pytest.raises(ValueError,match="score disagrees"):
        analyze_observed(dev,[r],"tampered")


def test_same_case_comparison_only_observed_cases(dev):
    ref=[observed(c) for c in dev[:3]]
    cand=[observed(dev[0],"inconclusive"),observed(dev[1])]
    summary=paired_comparison(ref,cand,dev,"test")
    assert summary["identical_case_count"]==2
    assert summary["reference_correct"]==2
    assert summary["comparison_is_same_case_not_full_diagnostic"]
    with pytest.raises(ValueError,match="unmatched"):
        paired_comparison(ref,cand+[observed(dev[0])],dev,"duplicate")


@pytest.mark.parametrize("status,change,decision",[
    ("explicitly_consistent_correct",{}, "equivalent"),
    ("reported_precedence_mismatch",{}, "inconclusive"),
    ("reported_factor_mismatch_with_coherent_decision",{"provenance_valid":False},"inconclusive"),
    ("compound_or_inconsistent_mismatch",{"provenance_valid":False},"recheck_required"),
])
def test_future_factor_probe_scoring_classifies_explicit_errors(dev,status,change,decision):
    f=reference_factors(next(x for x in dev if x["semantic_root"]=="equivalent"))
    raw=f.to_dict();raw.update(change)
    out=score_explicit_factor_probe(f,{"factors":raw,"decision":decision})
    assert out["status"]==status


def test_factor_probe_rejects_incomplete_contract(dev):
    f=reference_factors(dev[0])
    assert score_explicit_factor_probe(f,{"decision":"equivalent"})["status"]=="invalid_probe_contract"
    raw=f.to_dict();raw.pop("effect")
    assert score_explicit_factor_probe(f,{"factors":raw,"decision":"equivalent"})["status"]=="invalid_probe_contract"
