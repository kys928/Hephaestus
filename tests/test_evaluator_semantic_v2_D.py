"""Read-only validation of the D-specific balanced early-stop screen."""
from collections import Counter
import copy
import json
from pathlib import Path

import pytest

from build_evaluator_semantic_repair_v2 import build_pack
from evaluator_semantic_v2_screen import screen_cases, screen_hash, verdict


@pytest.fixture(scope="module")
def sampled():
    frozen = build_pack("D")["partitions"]["evaluator"]["development_diagnostic"]
    selected = screen_cases(frozen)
    return frozen, selected


def scored(selected, wrong_ids=()):
    wrong = set(wrong_ids)
    return [
        {
            "case_id": case["case_id"],
            "score": {"decision_exact": case["case_id"] not in wrong},
        } for case in selected
    ]


def test_d_screen_is_64_balanced_subset_of_original_immutable_256(sampled):
    frozen, selected = sampled
    assert len(frozen) == 256 and len(selected) == 64
    assert len(set(r["case_id"] for r in selected)) == 64
    assert {r["case_id"] for r in selected} <= {r["case_id"] for r in frozen}
    assert Counter(r["expected"]["decision"] for r in selected) == {
        state: 8 for state in {r["expected"]["decision"] for r in frozen}
    }
    assert screen_hash(screen_cases(list(reversed(frozen)))) == screen_hash(selected)


def test_perfect_screen_is_not_certificate(sampled):
    _,selected=sampled
    outcome=verdict(selected, scored(selected))
    assert outcome["status"] == "screen_gate_reachable"
    assert outcome["scientific_improvement_demonstrated"] is False
    assert outcome["screen_is_not_certification"] is True


def test_thirteen_wrong_prove_overall_gate_unreachable(sampled):
    _,selected=sampled
    outcome=verdict(selected,scored(selected,{r["case_id"] for r in selected[:13]}))
    assert outcome["status"]=="screen_rejected"
    assert outcome["full_accuracy_upper_bound"] == (51+192)/256
    assert "overall_95pct_mathematically_unreachable" in outcome["unreachable_gates"]


def test_two_wrong_in_robust_state_prove_state_gate_unreachable(sampled):
    _,selected=sampled
    ids=[r["case_id"] for r in selected if r["expected"]["decision"]=="regressed"][:2]
    outcome=verdict(selected,scored(selected,ids))
    assert outcome["status"]=="screen_rejected"
    assert "state_gate_unreachable:regressed" in outcome["unreachable_gates"]


def test_four_wrong_in_weak_state_prove_state_gate_unreachable(sampled):
    _,selected=sampled
    ids=[r["case_id"] for r in selected if r["expected"]["decision"]=="equivalent"][:4]
    outcome=verdict(selected,scored(selected,ids))
    assert "state_gate_unreachable:equivalent" in outcome["unreachable_gates"]


def test_screen_rejects_missing_or_duplicate_samples(sampled):
    _,selected=sampled
    rows=scored(selected)
    with pytest.raises(ValueError):
        verdict(selected, rows[:-1])
    with pytest.raises(ValueError):
        verdict(selected, rows[:-1]+[copy.deepcopy(rows[0])])


def test_d_launcher_dry_run_routes_only_D_and_original_bootstrap():
    import launch_evaluator_semantic_repair_v2 as launcher
    from prepare_evaluator_semantic_repair_v2 import prepare
    cfg,_,_,manifest=prepare("D")
    body=launcher.request(cfg,"0"*40,"d-dryrun","evaluate",real_env=False,arm="D")
    flat=json.dumps(body)
    assert "run_evaluator_semantic_repair_v2.py" in flat
    assert "--arm D" in flat
    assert "--arm C" not in flat
    assert "run_evaluator_semantic_invariance_repair_v1.py" not in flat
    assert manifest["arm"] == "D"
