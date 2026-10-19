"""Fresh CPU-only single-variable experiment design and precedence oracle tests."""
from dataclasses import replace
from collections import Counter

import pytest
from hephaestus.schemas.evaluator_factors import combination_space
from hephaestus.policy.evaluator_semantic_oracle import scientific_state
from evaluator_effect_polarity_contrasts import (
    DEFAULT,RECHECK,TRANSITIONS,FACTOR_KEYS,EVIDENCE_KEYS,render,
    contrast_panel,validate_panel,inference_fixture_manifest,
)


def expected_precedence(f):
    if not f.gate_pass:return "scientific_rejection"
    if not f.complete:return "incomplete_evidence"
    if not f.provenance_valid:return "inconclusive"
    if not f.variance_stable:return "recheck_required"
    if f.effect=="negative":return "regressed"
    if f.effect=="neutral":return "equivalent"
    return "certification_ready" if f.recheck_satisfied else "improved"


def test_full_96_oracle_truth_table():
    states=Counter()
    for f in combination_space():
        states[scientific_state(f)]+=1
        assert scientific_state(f)==expected_precedence(f)
    assert len(combination_space())==96
    assert set(states)=={
        "scientific_rejection","incomplete_evidence","inconclusive",
        "recheck_required","regressed","equivalent","improved",
        "certification_ready"
    }


def test_fresh_60_pairs_change_exactly_one_scientific_variable():
    report=validate_panel()
    assert report["panel_size"]==60
    assert report["single_variable_pairs"]==60
    assert report["test_pairs_with_throughput"]==30
    assert report["per_transition_presentations"]==4
    assert report["frozen_development_examples_reused"] is False
    assert report["approval_for_model_inference"] is False


@pytest.mark.parametrize("name",TRANSITIONS)
def test_explicit_state_transition_is_consistent_with_oracle(name):
    factors,changed,value,target=TRANSITIONS[name]
    revised=replace(factors,**{changed:value})
    diffs=[key for key in FACTOR_KEYS
           if getattr(factors,key)!=getattr(revised,key)]
    assert diffs==[changed]
    assert scientific_state(revised)==target
    for distractor in (False,True):
        a=render(factors,distractor=distractor,phrasing="neutral_numbers")
        b=render(revised,distractor=distractor,phrasing="neutral_numbers")
        # A sign flip legitimately changes secondary throughput too, but
        # that is an explicitly declared NON-AUTHORITATIVE derived distractor.
        observed=[key for key in EVIDENCE_KEYS.values() if a[key]!=b[key]]
        assert observed==[EVIDENCE_KEYS[changed]]


def test_equivalent_versus_recheck_does_not_require_effect_change():
    equivalent=DEFAULT
    recheck=replace(equivalent,variance_stable=False)
    assert equivalent.effect==recheck.effect=="neutral"
    assert scientific_state(equivalent)=="equivalent"
    assert scientific_state(recheck)=="recheck_required"


def test_recheck_precedence_masks_effect_sign_and_confirmation():
    assert scientific_state(RECHECK)=="recheck_required"
    assert scientific_state(replace(RECHECK,effect="negative"))=="recheck_required"
    assert scientific_state(replace(RECHECK,effect="neutral"))=="recheck_required"
    assert scientific_state(replace(RECHECK,recheck_satisfied=True))=="recheck_required"


def test_invalid_provenance_masks_positive_effect_and_unstable_variance():
    f=replace(DEFAULT,provenance_valid=False,effect="positive",variance_stable=False)
    assert scientific_state(f)=="inconclusive"
    assert scientific_state(replace(f,effect="negative"))=="inconclusive"
    assert scientific_state(replace(f,variance_stable=True))=="inconclusive"


def test_reference_positive_effect_is_not_a_positive_final_state():
    for gate,complete,prov,stable,state in [
        (False,True,True,True,"scientific_rejection"),
        (True,False,True,True,"incomplete_evidence"),
        (True,True,False,True,"inconclusive"),
        (True,True,True,False,"recheck_required")]:
        f=replace(DEFAULT,gate_pass=gate,complete=complete,
                  provenance_valid=prov,variance_stable=stable,effect="positive")
        assert scientific_state(f)==state
        assert f.effect=="positive"


def test_negative_throughput_does_not_reverse_positive_primary_quality():
    f=replace(DEFAULT,effect="positive")
    no_distractor=render(f,distractor=False,phrasing="neutral_numbers")
    distraction=render(f,distractor=True,phrasing="neutral_numbers")
    assert no_distractor["primary_quality_delta"]==distraction["primary_quality_delta"]
    assert "report-only throughput delta: -1.0" in distraction["secondary_throughput_delta"]
    assert scientific_state(f)=="improved"


def test_fresh_panel_manifest_has_no_gold_labels_in_prompts():
    m=inference_fixture_manifest()
    assert m["prompt_count"]==120
    assert len(m["unlabeled_prompt_identity_sha256"])==64
    assert m["authorized_inference"] is False
    assert m["paid_run"] is False
    assert m["precedence_invariant_pairs"]>=8


def test_validator_rejects_multiple_changed_factors():
    rows=contrast_panel()
    tampered=dict(rows[0])
    tampered["modified_factors"]=replace(rows[0]["modified_factors"],complete=False)
    with pytest.raises(ValueError,match="multiple scientific factors"):
        validate_panel([tampered]+rows[1:])
