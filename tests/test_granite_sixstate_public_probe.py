"""No-model no-network checks for the proposed six-state public diagnostic."""
from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from hephaestus.development.granite_sixstate_public_probe import (
    ORIGINAL_ADAPTER, ORIGINAL_MODEL_ID, ORIGINAL_REVISION, PRODUCTION_ONTOLOGY,
    STATE_ACTIONS, build_proposed_pack, canonical_hash, oracle,
    validate_proposed_pack, DEFAULT,
)

ROOT = Path(__file__).resolve().parents[1]


def test_pure_proposed_public_pack_balanced_and_traceable():
    first = build_proposed_pack()
    second = build_proposed_pack()
    assert first == second
    assert canonical_hash(first) == canonical_hash(second)
    assert first["case_count"] == 120
    assert first["ontology"] == PRODUCTION_ONTOLOGY
    assert Counter(x["reference"]["decision"] for x in first["cases"]) == Counter({
        name: 20 for name in STATE_ACTIONS
    })
    assert first["requires_independent_design_acceptance"] is True
    assert first["research_execution_authorized"] is False
    assert first["no_model_outputs_consumed"] is True


def test_frozen_registry_parent_identity_is_fixed():
    registry = json.loads((ROOT / "configs/models/hephaestus_role_model_stack_v1.json").read_text())
    target = registry["roles"]["evaluator"]
    assert target["model_id"] == ORIGINAL_MODEL_ID
    assert target["revision"] == ORIGINAL_REVISION
    assert target["adapter"]["sha256"] == ORIGINAL_ADAPTER


def test_every_causal_pair_changes_only_declared_factor_and_decision():
    pack = build_proposed_pack()
    groups = {}
    for case in pack["cases"]:
        groups.setdefault(case["pair_id"], []).append(case)
    causal = [pair for pair in groups.values() if pair[0]["pair_kind"] == "causal_single_factor"]
    nuisance = [pair for pair in groups.values() if pair[0]["pair_kind"] == "irrelevant_single_factor"]
    assert len(causal) == 48 and len(nuisance) == 12
    for a, b in causal:
        changes = [name for name in DEFAULT if a["oracle_factors"][name] != b["oracle_factors"][name]]
        assert len(changes) == 1
        assert changes == [a["changed_field"]]
        assert a["reference"]["decision"] != b["reference"]["decision"]
        assert a["evidence"][-1]["fact"] == b["evidence"][-1]["fact"]
    for a, b in nuisance:
        assert a["oracle_factors"] == b["oracle_factors"]
        assert a["reference"] == b["reference"]
        assert a["evidence"][-1]["fact"] != b["evidence"][-1]["fact"]


@pytest.mark.parametrize("changed,value,expected", [
    ("evaluation_protocol", "mismatched", "evaluation_invalid"),
    ("checkpoint_provenance", "unverifiable", "evaluation_invalid"),
    ("mandatory_gates", "failed", "scientific_rejection"),
    ("sample_coverage", "incomplete", "incomplete_evidence"),
    ("repeatability", "unstable", "incomplete_evidence"),
    ("quality_delta", "negative", "regressed"),
    ("quality_delta", "neutral", "no_material_change"),
    ("quality_delta", "positive", "improved"),
])
def test_proposed_oracle_for_singular_factors(changed, value, expected):
    assert oracle({**DEFAULT, changed: value}) == expected


def test_proposed_precedence_is_explicit_and_unchanged_by_optimistic_distractor():
    factors = {**DEFAULT, "mandatory_gates": "failed",
               "sample_coverage": "incomplete", "quality_delta": "positive"}
    assert oracle(factors) == "scientific_rejection"
    assert oracle({**factors, "checkpoint_provenance": "unverifiable"}) == "evaluation_invalid"


def test_no_protected_partitions_and_no_certification_outputs_in_pack():
    pack = build_proposed_pack()
    text = json.dumps(pack, sort_keys=True)
    assert '"partition": "public_development_proposal"' in text
    assert "protected_holdout" not in text
    assert "certification.jsonl" not in text
    assert "raw_generation" not in text
    assert all("expected_decision" not in case["situation"] for case in pack["cases"])
    assert all(not any("expected_decision" in x["fact"] for x in case["evidence"])
               for case in pack["cases"])


def test_mutating_one_expected_decision_rejected():
    pack = build_proposed_pack()
    pack["cases"][0]["reference"]["decision"] = "improved"
    with pytest.raises(ValueError, match="oracle label inconsistent"):
        validate_proposed_pack(pack)


def test_removing_one_pair_denied():
    pack = build_proposed_pack()
    pack["cases"].pop()
    with pytest.raises(ValueError, match="120 cases"):
        validate_proposed_pack(pack)


def test_wrong_case_ontology_denied():
    pack = build_proposed_pack()
    pack["cases"][0]["ontology"] = "historical-semantic-invariance-evaluator.v1"
    with pytest.raises(ValueError, match="cross-ontology"):
        validate_proposed_pack(pack)


def test_unsupervised_irrelevant_input_does_not_change_oracle():
    pack = build_proposed_pack()
    pair = [x for x in pack["cases"] if x["pair_id"] == "NUISANCE-improved-00"]
    assert len(pair) == 2
    assert pair[0]["reference"]["decision"] == pair[1]["reference"]["decision"]
    assert pair[0]["evidence"][-1]["fact"] != pair[1]["evidence"][-1]["fact"]


def test_cannot_smuggle_an_extra_factor_into_oracle():
    with pytest.raises(ValueError):
        oracle({**DEFAULT, "undeclared": "fake"})


def test_repeated_build_does_not_need_model_or_external_provider(monkeypatch):
    # All inputs are literal authored public task propositions; this program
    # does not read remote models, invoke S3, or load any checkpoint.
    assert build_proposed_pack()["case_count"] == 120
