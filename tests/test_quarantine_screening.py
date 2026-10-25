"""Synthetic adversarial checks; these tests establish no real source clearance."""
import base64
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hephaestus.data.quarantine_screening import pii_counts, duplicate_counts, public_probe_overlap
from hephaestus.policy.custodial_references import verify_custodial_reference_bundle
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.verified import canonical_bytes

def test_source_pii_checks_excluded_supervision_fields_without_returning_text():
    report = pii_counts([{"question": "Safe?", "provided_answer": "hidden@example.org"}])
    assert report["candidate_counts"]["email"] == 1
    assert report["rows_flagged"] == 1
    assert "hidden@" not in json.dumps(report)
    assert report["status"] == "human_review_required"


def test_duplicates_normalize_without_deleting_source_or_claiming_semantic_pass():
    report = duplicate_counts(["Alpha   beta", "alpha beta", "alpha beta", " "])
    assert report["exact_repeated_rows"] == 2
    assert report["nonempty_rows"] == 3
    assert report["semantic_contamination_checked"] is False
    assert report["exhaustive_near_duplicate_search"] is False


def test_public_overlap_is_partial_even_if_no_hits():
    report = public_probe_overlap(["no overlap whatsoever"], ["one two three four five six"])
    assert report["rows_with_prompt_substrings"] == 0
    assert report["sealed_holdout_checked"] is False
    assert report["status"] == "partial_reference_coverage"
    assert public_probe_overlap(["prefix one two three four five six suffix"],
        ["one two three four five six"])["rows_with_prompt_substrings"] == 1


def _scene(tmp_path, change=None):
    pytest.importorskip("cryptography")  # mandatory installation in dedicated custody workflow
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    store = FileSystemArtifactStore(tmp_path)
    def put(value):
        return store.put_bytes(canonical_bytes(value)).artifact_ref
    private = Ed25519PrivateKey.generate()  # SYNTHETIC trust root, never production.
    refs = {}
    for role in ("training", "development", "diagnostic", "holdout"):
        manifest = {"schema": "custodial-reference-manifest.v1", "role": role,
            "custodian_id": "synthetic-test-custodian", "row_count": 10,
            "content_ref": "sha256:" + hashlib.sha256(role.encode()).hexdigest(),
            "sealed_at": "2026-10-24T00:00:00Z", "normalization_profile": "original-text-v1",
            "original_text_available_to_custodian": True, "exposure_inventory_complete": True}
        if change == "incomplete_inventory" and role == "training":
            manifest["exposure_inventory_complete"] = False
        refs[role] = put(manifest)
    scope = put({"schema": "custodial-contamination-audit-scope.v1", "training_allowed": False,
        "protected_reference_access_allowed": change != "source_only_scope",
        "execution_boundary": "custodian_environment", "reference_manifest_refs": refs,
        "source_file_hashes": {"synthetic/train.jsonl": "sha256:" + "a" * 64},
        "purposes": ["exact_near_duplicate", "semantic_contamination", "frozen_evaluation_leakage"]})
    statement = {"schema": "custodial-reference-access.v1", "custodian_id": "synthetic-test-custodian",
        "reference_manifest_refs": refs, "scope_ref": scope, "training_allowed": False,
        "execution_boundary": "custodian_environment", "original_text_audits_authorized": True,
        "issued_at": "2026-10-25T00:00:00Z", "expires_at": "2026-10-26T00:00:00Z"}
    if change == "training":
        statement["training_allowed"] = True
    if change == "expired":
        statement["expires_at"] = "2026-10-25T01:00:00Z"
    statement_ref = put(statement)
    signature = private.sign(canonical_bytes(statement))
    if change == "signature":
        signature = b"x" * 64
    envelope = put({"schema": "custodial-reference-signature.v1", "statement_ref": statement_ref,
        "signature_base64": base64.b64encode(signature).decode()})
    return store, dict(signature_ref=envelope, trusted_statement_ref=statement_ref,
        trusted_public_key=private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw),
        trusted_custodian_id="synthetic-test-custodian", expected_scope_ref=scope,
        expected_reference_refs=refs, now=datetime(2026, 10, 25, 12, tzinfo=timezone.utc))


def test_exact_external_custody_bundle_verifies_synthetic_signature(tmp_path):
    store, args = _scene(tmp_path)
    assert len(verify_custodial_reference_bundle(store, **args)) == 4


@pytest.mark.parametrize("change", ["training", "expired", "signature", "incomplete_inventory", "source_only_scope"])
def test_invalid_custody_never_admitted(tmp_path, change):
    store, args = _scene(tmp_path, change)
    with pytest.raises(PermissionError):
        verify_custodial_reference_bundle(store, **args)


@pytest.mark.parametrize("mutation", ["scope", "statement", "reference", "key"])
def test_substituted_trust_roots_refused(tmp_path, mutation):
    store, args = _scene(tmp_path)
    if mutation == "scope":
        args["expected_scope_ref"] = "sha256:" + "b" * 64
    elif mutation == "statement":
        args["trusted_statement_ref"] = "sha256:" + "b" * 64
    elif mutation == "reference":
        args["expected_reference_refs"] = dict(args["expected_reference_refs"], holdout="sha256:" + "b" * 64)
    else:
        args["trusted_public_key"] = b"y" * 32
    with pytest.raises(PermissionError):
        verify_custodial_reference_bundle(store, **args)


def test_screening_scope_requires_exact_digest_and_no_training():
    path = Path(__file__).parents[1] / "scripts/run_quarantine_source_screening.py"
    spec = importlib.util.spec_from_file_location("quarantine_driver", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    scope = json.loads(driver.SCOPE.read_text())
    driver.verify_scope(scope, driver.digest(canonical_bytes(scope)))
    with pytest.raises(PermissionError):
        driver.verify_scope(scope, "sha256:" + "0" * 64)
    scope["training_allowed"] = True
    with pytest.raises(PermissionError):
        driver.verify_scope(scope, driver.digest(canonical_bytes(scope)))


def test_measurement_profile_preserves_original_prompt_target_and_line_boundaries():
    path = Path(__file__).parents[1] / "scripts/run_quarantine_source_screening.py"
    spec = importlib.util.spec_from_file_location("quarantine_driver", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    row = {"question": "What?", "reference_answer": "First line.\nSecond line."}
    assert driver.mapped_profile_text(row, is_saf=True) == "What?\nFirst line.\nSecond line."
    assert driver.mapped_profile_text({"text": "A\nB"}, is_saf=False) == "A\nB"
