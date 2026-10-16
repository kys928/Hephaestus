"""CPU-only factor probe protocol, no model loads, pod access, or S3."""
import json
from collections import Counter

import pytest

from evaluator_semantic_factor_probe_protocol import (
    panel,plan_manifest,parse_factor_json,evaluate_reports,
    D_CHECKPOINT_SHA256,MAX_NEW_TOKENS,
)
from hephaestus.schemas.evaluator_factors import ScientificFactors


@pytest.fixture(scope="module")
def records():
    return panel()


def future_rows(records, *, incorrect_factor=None):
    out=[]
    for r in records:
        f=r["reference_factors"].to_dict()
        if incorrect_factor is not None and r["target_state"]=="equivalent":
            f[incorrect_factor]=False
        out.append({"case_id":r["case_id"],
                    "prompt_sha256":r["prompt_sha256"],
                    "checkpoint_sha256":D_CHECKPOINT_SHA256,
                    "raw_factors":json.dumps(f)})
    return out


def paired_saved_decisions(records):
    return [{"case_id":r["case_id"],
             "score":{"parsed":{"decision":r["target_state"]}}} for r in records]


def test_panel_is_original_D_screen_without_target_leak(records):
    assert len(records)==64
    assert Counter(r["target_state"] for r in records)=={
        state:8 for state in {r["target_state"] for r in records}}
    for r in records:
        text=r["prompt"]
        assert r["case_id"] not in text
        assert '"expected"' not in text
        assert '"semantic_root"' not in text
        assert '"target_rationale"' not in text


def test_manifest_declares_no_execution_or_new_spend(records):
    m=plan_manifest(records)
    assert m["new_spending_authorized"] is False
    assert m["runpod_allocation_allowed"] is False
    assert m["case_count"]==64
    assert m["max_new_tokens"]==MAX_NEW_TOKENS
    assert len(m["prompt_identity_sha256"])==64
    assert "prompt" not in m and "reference_factors" not in m


def test_strict_factor_json_refuses_duplicates_nan_wrong_types():
    valid={"gate_pass":True,"complete":True,"provenance_valid":True,
           "effect":"neutral","variance_stable":True,"recheck_satisfied":False}
    assert parse_factor_json(json.dumps(valid))==ScientificFactors(**valid)
    assert parse_factor_json('{"gate_pass":true,"gate_pass":false}') is None
    assert parse_factor_json(json.dumps({**valid,"unexpected":True})) is None
    assert parse_factor_json(json.dumps({**valid,"complete":"yes"})) is None
    assert parse_factor_json(json.dumps({**valid,"effect":"unknown"})) is None
    assert parse_factor_json("fenced "+json.dumps(valid)) is None


def test_perfect_explicit_factors_and_saved_correct_decisions(records):
    report=evaluate_reports(records,future_rows(records),paired_saved_decisions(records))
    assert report["total"]==64
    assert report["all_six_factors_correct"]==64
    assert report["saved_D_decisions_correct"]==64
    assert report["both_factors_and_D_decision_correct"]==64
    assert report["per_factor"]["provenance_valid"]["correct"]==64


def test_explicit_wrong_provenance_for_equivalent_class_separated(records):
    rs=future_rows(records,incorrect_factor="provenance_valid")
    old=paired_saved_decisions(records)
    report=evaluate_reports(records,rs,old)
    assert report["all_six_factors_correct"]==56
    assert report["per_factor"]["provenance_valid"]["correct"]==56
    assert report["per_factor"]["variance_stable"]["correct"]==64
    assert report["by_state"]["equivalent"]["all_factors_correct"]==0
    assert report["observable_mismatch_categories"]["compound_or_inconsistent_mismatch"]==8


def test_missing_duplicate_or_checkpoint_drift_rejected(records):
    rs=future_rows(records);old=paired_saved_decisions(records)
    with pytest.raises(ValueError, match="incomplete/duplicated"):
        evaluate_reports(records,rs[:-1],old)
    with pytest.raises(ValueError,match="incomplete/duplicated"):
        evaluate_reports(records,rs[:-1]+[rs[0]],old)
    with pytest.raises(ValueError,match="lineage mismatch"):
        wrong=[dict(x) for x in rs];wrong[0]["checkpoint_sha256"]="0"*64
        evaluate_reports(records,wrong,old)
    with pytest.raises(ValueError,match="lineage mismatch"):
        wrong=[dict(x) for x in rs];wrong[0]["prompt_sha256"]="0"*64
        evaluate_reports(records,wrong,old)
