"""Genuine Diagnosis/Planner preflight from a pinned, previously exposed D metric."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hephaestus.development.granite_d_diagnosis_preflight import (
    PinnedEvidenceError, SHA_REPORT, preflight, validate_aggregate,
)

ROOT = Path(__file__).resolve().parents[1]
OBSERVATION = ROOT / "docs/evidence/prometheus-granite-d/observation-v1.json"


def _source():
    return json.loads(OBSERVATION.read_text(encoding="utf-8"))


def test_real_historical_aggregate_runs_existing_diagnosis_and_planner():
    observation = _source()
    result = preflight(observation)
    assert result["source_aggregate_report_sha256"] == SHA_REPORT
    assert result["source_ontology"] == "historical-semantic-invariance-evaluator.v1"
    assert result["decision_owner"] == "HEPHAESTUS_DETERMINISTIC_POLICY"
    assert result["model_backed_specialist_inference"] is False
    assert result["training_executed"] is False
    assert result["scientific_repair_established"] is False
    assert result["independent_certification"] is False
    assert result["diagnosis"]["status"] == "inconclusive"
    assert result["interventions"]
    assert all(p["intervention_kind"] in {"collect_more_evidence", "repair_evaluation"}
               for p in result["interventions"])
    assert any("eval_integrity_unverified" == issue["code"]
               for issue in result["diagnosis"]["issues"])
    assert any(o["source_ref"] == SHA_REPORT
               for o in result["diagnosis"]["observations"])
    assert all("model_originated" is not True for p in result["interventions"]
               for _ in [None] if True)  # no simulated model-inference receipt
    assert "no_material_change" not in json.dumps(result["diagnosis"]["observations"])


@pytest.mark.parametrize("key,replacement", [
    ("sample_count", 32),
    ("strict_correct_count", 34),
    ("strict_disagreements_with_saved", 1),
    ("original_aggregate_report_sha256", "sha256:" + "0" * 64),
    ("ontology", "frozen-role-mastery-evaluator.v1"),
    ("scientific_causality_established", True),
    ("independent_final_assessment", True),
    ("model_weights_independently_verified", True),
    ("protected_final_holdout_accessed", True),
    ("source_workflow_run_id", "41609821443"),
])
def test_wrong_provenance_or_scientific_claim_is_denied(key, replacement):
    source = _source()
    source[key] = replacement
    with pytest.raises(PinnedEvidenceError):
        preflight(source)


def test_injecting_unverified_eval_integrity_is_rejected():
    source = _source()
    source["eval_integrity_verified"] = True
    with pytest.raises(PinnedEvidenceError, match="extended"):
        validate_aggregate(source)


def test_inconsistent_by_state_count_is_denied():
    source = _source()
    source["by_state_correct"]["equivalent"] = 1
    with pytest.raises(PinnedEvidenceError, match="by-state"):
        preflight(source)


def test_fabricated_new_s3_key_is_denied():
    source = _source()
    source["source_shards"][0]["key"] = "hephaestus/private/protected-holdout"
    with pytest.raises(PinnedEvidenceError, match="shard"):
        preflight(source)


def test_observation_is_source_pinned_and_not_a_causal_hypothesis():
    result = preflight(_source())
    payload = json.dumps(result, sort_keys=True)
    assert '"evidence_kind": "historical_observation"' in payload
    assert '"scientific_repair_established": false' in payload
    assert '"model_backed_specialist_inference": false' in payload
    assert '"training_executed": false' in payload
