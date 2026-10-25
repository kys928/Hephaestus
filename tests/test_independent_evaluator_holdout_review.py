"""Signed external-custody review checks: synthetic inputs are NOT certificates."""
from __future__ import annotations

import base64
import hashlib
from dataclasses import replace

import pytest

from hephaestus.evaluation.independent_holdout_review import review_independent_evaluator_holdout
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.verified import canonical_bytes, read_verified_json


STATES = (
    "equivalent", "recheck_required", "regressed", "scientific_rejection",
    "improved", "incomplete_evidence", "inconclusive", "not_passed",
)


def _put(store, item):
    return store.put_bytes(canonical_bytes(item), media_type="application/json").artifact_ref


def _fp(word: str) -> str:
    return "sha256:" + hashlib.sha256(word.encode()).hexdigest()


def _scene(tmp_path, *, regress=False, overlap=None, wins=10):
    crypto = pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    store = FileSystemArtifactStore(tmp_path / "immutable")
    cases = []
    for i in range(24):
        cases.append({
            "case_id": f"secret-case-{i}",
            "gold_state": STATES[i % len(STATES)],
            "contrast_id": f"independent-pair-{i//2}",
            "protected": i < 6,
            "example_fingerprint": _fp(f"independent-heldout-example-{i}"),
        })
    pack = _put(store, {"schema": "scientific-state-holdout.v1",
                        "split": "sealed_holdout", "cases": cases})
    diagnostics = _put(store, {"schema": "scientific-diagnostic-cases.v1",
                               "case_ids": ["known-diagnostic-A"]})
    exposure_train = _put(store, {
        "schema": "scientific-training-exposure.v1",
        "case_ids": ["training-only"] if overlap != "training_id" else ["secret-case-7"],
        "example_fingerprints": [_fp("training-only")] if overlap != "training_hash"
                                 else [_fp("independent-heldout-example-7")],
    })
    exposure_dev = _put(store, {
        "schema": "scientific-development-exposure.v1",
        "case_ids": ["development-only"],
        "example_fingerprints": [_fp("development-only")],
    })
    settings = _put(store, {"decoding": "greedy", "max_new_tokens": 128,
                            "sampling_seeded": True})
    baseline_refs = []
    candidate_refs = []
    for seed in (11, 29, 47):
        def prediction(kind, checkpoint, rows):
            return _put(store, {
                "schema": "scientific-state-predictions.v1",
                "pack_ref": pack, "run_id": kind,
                "checkpoint_ref": checkpoint, "seed": seed,
                "settings_ref": settings,
                "predictions": [{"case_id": cases[i]["case_id"], "state": state}
                                for i, state in enumerate(rows)],
            })
        baseline_rows = []
        candidate_rows = []
        for i, item in enumerate(cases):
            # All protected pairs remain good. The remaining 10 pairs are
            # candidate-improved in the complete fixture; configurable loss.
            baseline_rows.append(item["gold_state"] if i < 4 else "wrong-state")
            pair = i // 2
            candidate_rows.append(item["gold_state"]
                                  if pair < wins + 2 else "wrong-state")
        if regress:
            candidate_rows[0] = "wrong-state"
        baseline_refs.append(prediction("frozen-parent", "checkpoint://parent", baseline_rows))
        candidate_refs.append(prediction("candidate-LoRA", "checkpoint://candidate", candidate_rows))
    witness = {
        "schema": "independent-holdout-witness.v1",
        "custodian_id": "separate-custodian-service",
        "holdout_ref": pack, "diagnostic_manifest_ref": diagnostics,
        "training_exposure_ref": exposure_train,
        "development_exposure_ref": exposure_dev,
        "baseline_prediction_refs": baseline_refs,
        "candidate_prediction_refs": candidate_refs,
        "baseline_checkpoint_ref": "checkpoint://parent",
        "candidate_checkpoint_ref": "checkpoint://candidate",
        "holdout_sealed_at": "2026-09-01T13:00:00+00:00",
        "training_started_at": "2026-10-01T13:00:00+00:00",
        "holdout_not_disclosed_to_training": True,
        "holdout_labels_blinded_until_predictions_fixed": True,
        "predictions_observed_by_independent_evaluator": True,
        "checkpoint_bytes_verified_by_independent_evaluator": True,
    }
    statement_ref = _put(store, witness)
    envelope = _put(store, {
        "schema": "independent-holdout-signature.v1",
        "statement_ref": statement_ref,
        "ed25519_signature_base64": base64.b64encode(
            private.sign(canonical_bytes(witness))).decode("ascii"),
    })
    args = {
        "approved_pack_ref": pack, "pre_registered_pack_ref": pack,
        "diagnostic_manifest_ref": diagnostics,
        "training_exposure_ref": exposure_train,
        "development_exposure_ref": exposure_dev,
        "baseline_refs": baseline_refs, "candidate_refs": candidate_refs,
        "attestation_ref": envelope,
        "trusted_custodian_id": "separate-custodian-service",
        "trusted_custodian_public_key": public,
        "expected_baseline_checkpoint_ref": "checkpoint://parent",
        "expected_candidate_checkpoint_ref": "checkpoint://candidate",
        "minimum_accuracy": 0.9, "minimum_contrast_accuracy": 0.9,
        "minimum_per_state_accuracy": 0.8, "minimum_matched_seeds": 3,
        "minimum_practical_gain": 0.1,
        "minimum_decisive_contrast_pairs": 8,
        "maximum_pair_p_value": 0.05,
    }
    return store, args, witness, private


def test_independent_signature_and_paired_effect_allow_review_but_never_certify(tmp_path):
    store, args, witness, private = _scene(tmp_path)
    result = review_independent_evaluator_holdout(store, **args)
    assert result.eligible_for_external_certification_review
    assert not result.blocking_issues
    assert result.winning_contrast_pairs == 10
    assert result.losing_contrast_pairs == 0
    assert result.independent_pair_sign_test_p < .01
    receipt = read_verified_json(store, result.review_receipt_ref)
    assert receipt["certification_state"] == "not_certified"
    assert receipt["live_activation_authorized"] is False
    assert receipt["assessment_ref"] == result.assessment.artifact_ref
    assessment = read_verified_json(store, result.assessment.artifact_ref)
    assert assessment["certification_state"] == "not_certified"


@pytest.mark.parametrize("mutation", [
    "wrong_pubkey", "unknown_custodian", "untrusted_pack",
    "training_overlap_id", "training_overlap_content",
    "changed_checkpoint", "unsigned_substitution",
])
def test_untrusted_or_leaked_holdout_cannot_be_admitted(tmp_path, mutation):
    store, args, witness, private = _scene(
        tmp_path,
        overlap=("training_id" if mutation == "training_overlap_id" else
                 "training_hash" if mutation == "training_overlap_content" else None))
    if mutation == "wrong_pubkey":
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
        args["trusted_custodian_public_key"] = Ed25519PrivateKey.generate().public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    elif mutation == "unknown_custodian":
        args["trusted_custodian_id"] = "self-appointed-reviewer"
    elif mutation == "untrusted_pack":
        args["pre_registered_pack_ref"] = _put(store, {
            "schema": "scientific-state-holdout.v1", "split": "sealed_holdout", "cases": []})
    elif mutation == "changed_checkpoint":
        args["expected_candidate_checkpoint_ref"] = "checkpoint://other-candidate"
    elif mutation == "unsigned_substitution":
        fake_witness = {**witness, "holdout_not_disclosed_to_training": False}
        replacement = _put(store, fake_witness)
        args["attestation_ref"] = _put(store, {
            "schema": "independent-holdout-signature.v1",
            "statement_ref": replacement,
            "ed25519_signature_base64": read_verified_json(
                store, args["attestation_ref"])["ed25519_signature_base64"],
        })
    with pytest.raises((ValueError, PermissionError)):
        review_independent_evaluator_holdout(store, **args)


def test_protected_regression_forces_review_refusal_even_if_other_pairs_improve(tmp_path):
    store, args, _, _ = _scene(tmp_path, regress=True)
    assessment = review_independent_evaluator_holdout(store, **args)
    assert not assessment.eligible_for_external_certification_review
    assert "protected_regression_detected" in assessment.blocking_issues
    record = read_verified_json(store, assessment.review_receipt_ref)
    assert record["certification_state"] == "not_certified"
    assert record["live_activation_authorized"] is False


def test_non_significant_contrast_effect_is_not_confused_with_improvement(tmp_path):
    store, args, _, _ = _scene(tmp_path, wins=4)
    result = review_independent_evaluator_holdout(store, **args)
    assert not result.eligible_for_external_certification_review
    assert ("insufficient_independent_decisive_contrast_pairs" in result.blocking_issues
            or "paired_sign_test_not_significant" in result.blocking_issues)
