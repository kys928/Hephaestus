"""Research precommit, denominator, source-custody and missingness contract tests.

All inputs here are synthetic fixtures: passing does not imply Diagnostician
inference, validated new source evidence or scientific model performance.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from hephaestus.development.diagnosis_calibration_protocol import (
    ADAPTER_SHA, ARMS, MODEL, REVISION, SEEDS, SLOTS, CalibrationProtocolError,
    canonical, fixed_controls, load_protocol, score_contract_fixtures,
    source_coverage, verify_raw_trial, frozen_signal_rule,
)
from hephaestus.diagnosis.service import EvidenceBasedDiagnosisService
from hephaestus.schemas.diagnosis_contract import DiagnosisRequest
from hephaestus.llm.role_contract import RoleMasteryContract


CONFIG = (Path(__file__).resolve().parents[1] /
          "configs/research/prometheus_diagnosis_evidence_contrast_v1.json")


def _labels():
    return {
        "M0":"inconclusive", "E1":"evaluation_integrity",
        "D1":"tokenizer", "C1":"inconclusive",
        "E2":"evaluation_integrity", "D2":"data_coverage",
        "N0":"inconclusive", "N1":"evaluation_integrity",
    }


def _fixtures():
    out = []
    for case_id in SLOTS:
        for arm in ARMS:
            for seed in SEEDS:
                out.append({
                    "case_id":case_id,"arm":arm,"seed":seed,
                    "decision":_labels()[case_id],"schema_valid":True,
                    "citation_exact":True,"citation_material":True,
                    "confidence":0.65,
                })
    return out


def _diagnose(*rows):
    return EvidenceBasedDiagnosisService().diagnose(DiagnosisRequest(
        "research-test","fixture-run","fixture-lineage","continuation_repair",
        observed_failures=list(rows)))


def test_protocol_is_precommitted_and_gpu_not_authorized():
    value, digest = load_protocol(CONFIG)
    assert len(digest) == 64
    assert value["sampling"]["invocation_ceiling"] == 48
    assert value["sampling"]["wallclock_hard_limit_seconds"] == 2700
    assert len(value["case_slots"]) == 8
    assert value["action_policy"]["planner_allowed_only_after_actual_supported_accepted_diagnosis"]
    assert not value["action_policy"]["training_and_adapter_changes_authorized"]
    assert not value["action_policy"]["granite_repair_authorized"]


def test_incomplete_or_self_declared_sources_never_enable_gpu():
    cfg, _ = load_protocol(CONFIG)
    initial = source_coverage(cfg, {})
    assert not initial["scientifically_admitted"]
    assert len(initial["missing_or_unverified"]) == 8
    forged = {"cases":{slot:{
        "source_kind":"authentic","source_refs":["sha256:" + "a"*64],
        "independent_verifier_receipts":["sha256:"+"b"*64, "sha256:"+"c"*64],
        "independent_label_receipt":"sha256:"+"d"*64,
    } for slot in SLOTS}}
    claimed = source_coverage(cfg, forged)
    assert len(claimed["possible_label_attestations"]) == 8
    assert not claimed["scientifically_admitted"]
    assert claimed["external_original_bytes_rehash_required"]
    assert not claimed["gpu_launch_authorized"]


def test_missing_eval_pack_and_scorecard_is_not_verified_evaluation_failure():
    report = _diagnose({
        "evidence_kind":"eval_report", "source_ref":"sha256:"+"a"*64,
        "eval_pack_integrity_level":"reference_only",
        "deterministic_scorecard":{},
    })
    assert report.status == "inconclusive"
    assert not any(h.failure_domain == "evaluation_integrity" for h in report.hypotheses)
    assert any(x.code == "eval_integrity_unverified" for x in report.issues)


def test_real_failure_signal_would_be_distinct_from_missingness_in_rules():
    report = _diagnose({
        "evidence_kind":"eval_report", "source_ref":"sha256:"+"b"*64,
        "eval_pack_integrity_level":"reference_only",
        "deterministic_scorecard":{},
        "signals":["eval_integrity_failed"],
    })
    assert any(h.failure_domain == "evaluation_integrity" and h.supporting_evidence_refs
               for h in report.hypotheses)
    # This tests a synthetic signal *contract*, not an independently observed failure.


def test_reference_label_can_never_be_changed_by_irrelevant_metadata():
    cfg,_ = load_protocol(CONFIG)
    labels = _labels()
    assert score_contract_fixtures(cfg, labels, _fixtures())["arms"][ARMS[0]][
        "nuisance_invariance"]["rate"] == 1.0
    labels["N1"] = "inconclusive"
    with pytest.raises(CalibrationProtocolError, match="metadata"):
        score_contract_fixtures(cfg, labels, _fixtures())


def test_denominators_and_negative_model_behaviors_are_visible():
    cfg, _ = load_protocol(CONFIG)
    result = score_contract_fixtures(cfg, _labels(), _fixtures())
    assert result["source"] == "synthetic_contract_fixture_only"
    assert not result["scientific_completion_claim"]
    for arm in ARMS:
        out = result["arms"][arm]
        assert out["supported_domain_precision"] == {"numerator":15,"denominator":15,"rate":1.0}
        assert out["appropriate_abstention"] == {"numerator":9,"denominator":9,"rate":1.0}
        assert out["evidence_sensitivity"]["denominator"] == 6
        assert out["nuisance_invariance"]["denominator"] == 6
        assert out["exact_and_material_citations"]["denominator"] == 24
        assert out["confidence_descriptive_only"]["mean"] == pytest.approx(0.65)
    corrupted = _fixtures()
    row = next(x for x in corrupted if x["case_id"]=="M0" and x["arm"]==ARMS[0])
    row["decision"] = "evaluation_integrity"
    row["confidence"] = 0.99
    r = score_contract_fixtures(cfg, _labels(), corrupted)["arms"][ARMS[0]]
    assert r["appropriate_abstention"]["rate"] < 1.0
    assert r["unsupported_domain_rate"]["numerator"] == 1
    assert r["evidence_sensitivity"]["rate"] < 1.0
    # High self-reported confidence is descriptive only, never proof of correctness.


def test_invalid_or_incomplete_result_is_not_best_case_imputed():
    cfg, _ = load_protocol(CONFIG)
    trials = _fixtures()[:-1]
    with pytest.raises(CalibrationProtocolError, match="incomplete"):
        score_contract_fixtures(cfg, _labels(), trials)
    trials = _fixtures()
    trials[0]["decision"] = None
    trials[0]["schema_valid"] = False
    trials[0]["citation_exact"] = False
    result = score_contract_fixtures(cfg, _labels(), trials)
    assert result["arms"][ARMS[0]]["invalid_output_rate"]["numerator"] == 1
    assert result["arms"][ARMS[0]]["exact_and_material_citations"]["numerator"] == 23
    with pytest.raises(CalibrationProtocolError, match="duplicate"):
        score_contract_fixtures(cfg, _labels(), trials + [trials[0]])


def test_fixed_controls_are_not_allowed_to_use_golden_labels():
    controls = fixed_controls(_labels())
    assert set(controls) == {"always_inconclusive","fixed_repair_evaluation",
                             "precommitted_signal_rule"}
    assert not controls["precommitted_signal_rule"]["uses_reference_labels"]
    assert controls["fixed_repair_evaluation"]["decisions"]["M0"] == "evaluation_integrity"


def test_forged_gpu_receipt_cannot_pass_sha_or_identity():
    cfg, _ = load_protocol(CONFIG)
    contract = RoleMasteryContract.load(
        Path(__file__).resolve().parents[1]/"configs/models/hephaestus_role_mastery_runtime_v1.json")
    answer = {
        "decision":"inconclusive", "action":"collect_more_evidence",
        "primary_variable":"evidence", "confidence":0.5,
        "evidence_refs":["sha256:" + "a"*64], "uncertainties":[], "rationale":"No cause verified.",
    }
    raw = json.dumps(answer)
    row = {
        "schema":"role-inference.v1", "role":"diagnosis",
        "generator":"PinnedTransformersRoleGenerator",
        "model_id":MODEL,"base_revision":REVISION,"adapter_sha256":ADAPTER_SHA,
        "seed":SEEDS[0],"evidence_refs":answer["evidence_refs"],
        "prompt":"fixture","prompt_sha256":__import__("hashlib").sha256(b"fixture").hexdigest(),
        "raw_output":raw,"output_sha256":__import__("hashlib").sha256(raw.encode()).hexdigest(),
        "status":"accepted","answer":answer,
        "execution":{"seed":SEEDS[0],"device":"cpu",
                     "gpu_name":"FAKE","generated_token_ids":[3]},
    }
    payload=canonical(row)
    with pytest.raises(CalibrationProtocolError, match="CUDA"):
        verify_raw_trial(payload,content_sha256=__import__("hashlib").sha256(payload).hexdigest(),
            expected_refs=set(answer["evidence_refs"]),expected_seed=SEEDS[0],contract=contract)
    with pytest.raises(CalibrationProtocolError, match="digest"):
        verify_raw_trial(payload,content_sha256="0"*64,
            expected_refs=set(answer["evidence_refs"]),expected_seed=SEEDS[0],contract=contract)



def test_observability_limit_counts_as_abstention_not_false_causal_pair():
    cfg, _ = load_protocol(CONFIG)
    trial = _fixtures()
    for item in trial:
        if item["case_id"] == "M0":
            item["decision"] = "observability_limit"
    scored = score_contract_fixtures(cfg, _labels(), trial)["arms"]
    for arm in ARMS:
        assert scored[arm]["appropriate_abstention"]["rate"] == 1.0
        assert scored[arm]["evidence_sensitivity"]["rate"] == 1.0
        # N0 still says inconclusive, so the literal decision changed despite
        # the same admissible evidence: the strict metadata invariance test fails.
        assert scored[arm]["nuisance_invariance"]["rate"] < 1.0



def test_fixed_heuristic_is_label_blind_and_cautious_about_missingness():
    assert frozen_signal_rule(set()) == "inconclusive"
    assert frozen_signal_rule({"eval_pack_unverified", "deterministic_scorecard_missing"}) == "inconclusive"
    assert frozen_signal_rule({"eval_integrity_failed"}) == "evaluation_integrity"
    assert frozen_signal_rule({"evaluation_settings_mismatch"}) == "evaluation_integrity"
    assert frozen_signal_rule({"tokenizer_mismatch"}) == "tokenizer"
    assert frozen_signal_rule({"data_coverage_gap"}) == "data_coverage"
    assert frozen_signal_rule({"eval_integrity_failed", "eval_integrity_verified"}) == "inconclusive"
    assert frozen_signal_rule({"eval_integrity_failed", "tokenizer_mismatch"}) == "inconclusive"
    assert frozen_signal_rule({"tokenizer_mismatch", "tokenizer_compatible"}) == "inconclusive"
    with pytest.raises(CalibrationProtocolError, match="signal set"):
        frozen_signal_rule(["eval_integrity_failed"])
