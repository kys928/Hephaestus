"""Prometheus decision-first contract tests; these are NOT real specialist inference."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from hephaestus.development.prometheus_scientific_decision import (
    D_ONTOLOGY, REF_D, REF_PARENT, PARENT_ONTOLOGY, PARENT_REVISION,
    PARENT_ADAPTER, PARENT_SHA, D_SHA, _verify, authenticated_observations,
    prepare, decide, _require_actual_gpu_receipt,
)
from hephaestus.diagnosis.pinned_historical import PinnedEvidenceError
from hephaestus.llm.certified_runtime import PinnedTransformersRoleGenerator


def _reports():
    parent = {
        "schema": "prometheus-original-certified-granite-dev-rescore.v1",
        "ontology": PARENT_ONTOLOGY,
        "original_model_revision": PARENT_REVISION,
        "original_selected_adapter_sha256": PARENT_ADAPTER,
        "partition": "previously_exposed_model_selection_development",
        "count": 200,
        "strict_projected_decision_correct": 200,
        "strict_raw_decision_correct": 200,
        "original_300_case_certification_not_read": True,
        "research_development_not_independent_final": True,
        "model_weights_newly_rehashed": False,
    }
    d = {
        "schema": "prometheus-granite-d-readonly-raw-rescore.v1",
        "scope": "historical_exposed_eight_state_development_only",
        "independent_strict_scoring": {
            "ontology": D_ONTOLOGY, "count": 64, "decision_correct": 33,
        },
        "model_weights_independently_verified": False,
        "independent_final_assessment": False,
    }
    return parent, d


def test_distinct_original_parent_and_historical_d_scopes():
    parent, d = _reports()
    observations, facts = authenticated_observations(parent, d)
    assert {r["source_ref"] for r in observations} == {REF_PARENT, REF_D}
    assert len(facts) == 2
    assert "ORIGINAL EXPOSED" in facts[0].fact
    assert "NOT directly matched" in facts[1].fact
    assert "33/64" in facts[1].fact and "200/200" in facts[0].fact
    assert all(x["evidence_kind"] == "historical_observation" for x in observations)
    assert not any("eval_integrity_verified" in x for x in observations)


def test_real_hephaestus_deterministic_diagnosis_only_admits_no_training():
    diagnosis, options, facts = prepare(*_reports())
    assert diagnosis.status == "inconclusive"
    assert any(x.code == "eval_integrity_unverified" for x in diagnosis.issues)
    assert diagnosis.metadata["historical_D_is_non_comparable"] is True
    assert options
    assert all(p.intervention_kind in {"collect_more_evidence", "repair_evaluation", "stop"}
               for p in options)
    assert all(p.intervention_kind not in {"resume_training", "replace_or_mix_dataset",
                                           "change_preprocessing", "change_model"}
               for p in options)
    assert {x.ref for x in facts} == {REF_PARENT, REF_D}


@pytest.mark.parametrize("source,key,value", [
    ("parent", "original_model_revision", "wrong"),
    ("parent", "original_selected_adapter_sha256", "0"*64),
    ("parent", "ontology", D_ONTOLOGY),
    ("parent", "count", 64),
    ("parent", "strict_raw_decision_correct", 199),
    ("parent", "original_300_case_certification_not_read", False),
    ("parent", "research_development_not_independent_final", False),
    ("parent", "model_weights_newly_rehashed", True),
    ("d", "scope", "protected_final"),
    ("d", "independent_final_assessment", True),
    ("d", "model_weights_independently_verified", True),
])
def test_source_provenance_and_exposure_claim_substitution_denied(source,key,value):
    parent, d = _reports()
    (parent if source == "parent" else d)[key] = value
    with pytest.raises(PinnedEvidenceError):
        prepare(parent,d)


def test_source_digest_mismatch_fail_closed_before_gpu(tmp_path: Path):
    file = tmp_path/"parent.json"
    file.write_text(json.dumps(_reports()[0]), encoding="utf-8")
    with pytest.raises(PinnedEvidenceError, match="SHA-256 mismatch"):
        _verify(file, PARENT_SHA)


def test_hardcoded_evidence_hashes_are_independent():
    assert len(PARENT_SHA) == 64 and len(D_SHA) == 64
    assert PARENT_SHA != D_SHA
    assert REF_PARENT.startswith("sha256:") and REF_D.startswith("sha256:")


def test_refuse_fake_role_model_without_any_cuda_or_s3(tmp_path: Path):
    parent, d = _reports()
    class FakeReasoner:
        generator = object()
    with pytest.raises(PermissionError, match="synthetic role generator"):
        decide(parent, d, reasoner=FakeReasoner(), out=tmp_path/"decision")
    assert not (tmp_path/"decision").exists()


def test_missing_cuda_provenance_never_counts_as_authenticated_science():
    class FakeReasoner:
        generator = object()
        artifact_store = object()
        contract = object()
    with pytest.raises((PermissionError, AttributeError, TypeError)):
        _require_actual_gpu_receipt(FakeReasoner(), "sha256:"+"0"*64, "planner")
