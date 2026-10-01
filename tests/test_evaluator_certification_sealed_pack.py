from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build_evaluator_certification_sealed_v1.py"
CANDIDATE = ROOT / "configs/experiments/hephaestus_evaluator_certification_candidate_v1.json"
EXPECTED_PACK_SHA = "1314d28c978538ca8bb5527c24e4200d5463f7ad90b084aaae1e656d0b530d25"


def _builder():
    spec = importlib.util.spec_from_file_location("evaluator_sealed_builder", BUILDER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sealed_pack_is_deterministic_fresh_balanced_and_disjoint() -> None:
    builder = _builder()
    first = builder.build_pack()
    second = builder.build_pack()
    builder.validate(first)
    builder.validate(second)
    assert builder.canonical_sha256(first) == EXPECTED_PACK_SHA
    assert builder.canonical_sha256(second) == EXPECTED_PACK_SHA

    part = first["partitions"]["evaluator"]
    assert len(part["preflight"]) == 8
    assert len(part["certification"]) == 128
    assert len(part["regression"]) == 128
    roots = set(builder.base.VOCAB["decision"])
    assert Counter(row["semantic_root"] for row in part["certification"]) == Counter({root: 16 for root in roots})
    assert Counter(row["semantic_root"] for row in part["regression"]) == Counter({root: 16 for root in roots})

    cert_contexts = {row["context"] for row in part["certification"]}
    regression_contexts = {row["context"] for row in part["regression"]}
    preflight_contexts = {row["context"] for row in part["preflight"]}
    assert cert_contexts.isdisjoint(regression_contexts)
    assert cert_contexts.isdisjoint(preflight_contexts)
    assert regression_contexts.isdisjoint(preflight_contexts)


def test_sealed_pack_has_opaque_identifiers_and_both_handoff_modes() -> None:
    builder = _builder()
    pack = builder.build_pack()
    builder.validate(pack)
    part = pack["partitions"]["evaluator"]
    for split in ("certification", "regression"):
        rows = part[split]
        assert any(row["kind"] == "rehearsal" and row["upstream_output"] is None for row in rows)
        assert any(row["kind"] == "interface" and row["upstream_output"] is not None for row in rows)
        for row in rows:
            assert row["context"].startswith("ctx_")
            assert all(ref.startswith("E-") for ref in row["allowed_evidence_refs"])
            identifiers = " ".join([row["case_id"], row["context"], *row["allowed_evidence_refs"]]).lower()
            assert not any(token in identifiers for token in builder.FORBIDDEN_IDENTIFIER_TOKENS)


def test_frozen_candidate_is_bound_to_exact_sealed_pack_and_no_paid_launch() -> None:
    cfg = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    assert cfg["candidate_status"] == "frozen_pending_fresh_sealed_certification"
    assert cfg["model"]["adapter_sha256"] == "913797ddb8d9d95f83d09a244e8efe430d7bfc4383e589499c2fcc2d943487ed"
    assert cfg["architecture"]["authoritative_primary_variable"] == "deterministic_from_scientific_state"
    assert cfg["architecture"]["fuzzy_evidence_reference_matching_allowed"] is False
    assert cfg["architecture"]["positive_auto_promotion_from_non_certification_state_allowed"] is False
    assert cfg["sealed_evidence"]["canonical_sha256"] == EXPECTED_PACK_SHA
    assert cfg["sealed_evidence"]["preflight_cases"] == 8
    assert cfg["sealed_evidence"]["certification_cases"] == 128
    assert cfg["sealed_evidence"]["regression_cases"] == 128
    assert cfg["sealed_evidence"]["status"] == "frozen_unseen"
    assert cfg["sealed_evidence"]["case_bodies_reviewed_after_freeze"] is False
    assert cfg["required_next_evidence"]["burned_diagnostic_reuse_forbidden"] is True
    assert cfg["governance"]["training_allowed"] is False
    assert cfg["governance"]["paid_certification_launch_allowed"] is False
    assert cfg["governance"]["production_promotion_allowed"] is False
