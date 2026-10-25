"""Independent held-out Evaluator review admission; deliberately NOT certification.

Requires a content-addressed, pre-registered holdout and an external Ed25519
custodian's signed custody/inference attestation. Recomputes scientific-state
metrics from immutable predictions. Neither this module nor its tests issue
certificates, authorize promotions, or change the active Evaluator.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from hephaestus.evaluation.scientific_state_certification import (
    ScientificStateAssessment, assess_scientific_states,
)
from hephaestus.storage.base import ArtifactStore
from hephaestus.storage.verified import canonical_bytes, read_verified_json


@dataclass(frozen=True, slots=True)
class IndependentHoldoutReview:
    assessment: ScientificStateAssessment
    review_receipt_ref: str
    eligible_for_external_certification_review: bool
    blocking_issues: tuple[str, ...]
    winning_contrast_pairs: int
    losing_contrast_pairs: int
    independent_pair_sign_test_p: float


def _time(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("independently attested timestamps must be ISO-8601 strings")
    try:
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid custody attestation timestamp") from exc
    if date.utcoffset() is None:
        raise ValueError("custody timestamps must carry a timezone")
    return date.astimezone(timezone.utc)


def _fingerprints(payload: dict, *, schema: str) -> tuple[set[str], set[str]]:
    if payload.get("schema") != schema:
        raise ValueError(f"incorrect {schema} exposure schema")
    case_ids = payload.get("case_ids")
    fingerprints = payload.get("example_fingerprints")
    if (not isinstance(case_ids, list) or not isinstance(fingerprints, list)
            or not all(isinstance(x, str) and x for x in case_ids)
            or not all(isinstance(x, str) and len(x) == 71
                       and x.startswith("sha256:")
                       and all(c in "0123456789abcdef" for c in x[7:])
                       for x in fingerprints)
            or len(case_ids) != len(set(case_ids))
            or len(fingerprints) != len(set(fingerprints))):
        raise ValueError("exposure manifest has missing, repeated or unpinned examples")
    return set(case_ids), set(fingerprints)


def _verify_signature(artifacts: ArtifactStore, *, attestation_ref: str,
                      trusted_public_key: bytes, trusted_custodian_id: str) -> dict:
    if len(trusted_public_key) != 32 or not trusted_custodian_id:
        raise ValueError("an externally provisioned Ed25519 custodian trust root is required")
    envelope = read_verified_json(artifacts, attestation_ref)
    if (envelope.get("schema") != "independent-holdout-signature.v1"
            or not isinstance(envelope.get("statement_ref"), str)):
        raise ValueError("independent holdout signature envelope missing")
    statement = read_verified_json(artifacts, envelope["statement_ref"])
    if (statement.get("schema") != "independent-holdout-witness.v1"
            or statement.get("custodian_id") != trusted_custodian_id):
        raise ValueError("independent custodian identity or schema mismatch")
    encoded = envelope.get("ed25519_signature_base64")
    if not isinstance(encoded, str):
        raise ValueError("independent signature is absent")
    try:
        signature = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("independent signature is malformed") from exc
    if len(signature) != 64:
        raise ValueError("invalid independent signature size")
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:
        raise RuntimeError("external signature verification unavailable: install cryptography") from exc
    try:
        Ed25519PublicKey.from_public_bytes(trusted_public_key).verify(
            signature, canonical_bytes(statement))
    except (InvalidSignature, ValueError) as exc:
        raise PermissionError("independent custody signature verification failed") from exc
    return statement


def _exact_sign_test(winning: int, losing: int) -> float:
    """One-sided paired sign test; ties excluded, contrast-pair is unit."""
    n = winning + losing
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(winning, n + 1)) / (2 ** n)


def review_independent_evaluator_holdout(
    artifacts: ArtifactStore, *, approved_pack_ref: str,
    pre_registered_pack_ref: str, diagnostic_manifest_ref: str,
    training_exposure_ref: str, development_exposure_ref: str,
    baseline_refs: Sequence[str], candidate_refs: Sequence[str],
    attestation_ref: str, trusted_custodian_id: str,
    trusted_custodian_public_key: bytes,
    expected_baseline_checkpoint_ref: str, expected_candidate_checkpoint_ref: str,
    minimum_accuracy: float, minimum_contrast_accuracy: float,
    minimum_per_state_accuracy: float = 0.5, minimum_matched_seeds: int = 3,
    minimum_practical_gain: float = 0.05,
    minimum_decisive_contrast_pairs: int = 8,
    maximum_pair_p_value: float = 0.05,
) -> IndependentHoldoutReview:
    """Admit reproducible evidence to external review, never certify or activate.

    All roots (including pre_registered_pack_ref and the Ed25519 public key)
    MUST be configured by an independent operator, not supplied by Planner,
    candidate inference, or training outputs. Signed witness assertions are
    checked, but signature verification cannot independently prove that a
    custodian told the truth or that semantically paraphrased leaks never
    occurred. Sign-test inference assumes independently sampled contrast pairs.
    """
    if (not approved_pack_ref.startswith("sha256:")
            or approved_pack_ref != pre_registered_pack_ref
            or not expected_baseline_checkpoint_ref
            or not expected_candidate_checkpoint_ref
            or expected_baseline_checkpoint_ref == expected_candidate_checkpoint_ref):
        raise PermissionError("independent preregistered holdout or checkpoint identity absent")
    if (type(minimum_practical_gain) not in (int, float)
            or not math.isfinite(minimum_practical_gain)
            or not 0 < minimum_practical_gain <= 1
            or type(maximum_pair_p_value) not in (int, float)
            or not math.isfinite(maximum_pair_p_value)
            or not 0 < maximum_pair_p_value <= 0.05
            or type(minimum_decisive_contrast_pairs) is not int
            or minimum_decisive_contrast_pairs < 8
            or type(minimum_matched_seeds) is not int
            or minimum_matched_seeds < 3):
        raise ValueError("independent preregistered evidence policy is invalid")

    witness = _verify_signature(
        artifacts, attestation_ref=attestation_ref,
        trusted_public_key=trusted_custodian_public_key,
        trusted_custodian_id=trusted_custodian_id)
    exact = {
        "holdout_ref": approved_pack_ref,
        "diagnostic_manifest_ref": diagnostic_manifest_ref,
        "training_exposure_ref": training_exposure_ref,
        "development_exposure_ref": development_exposure_ref,
        "baseline_prediction_refs": list(baseline_refs),
        "candidate_prediction_refs": list(candidate_refs),
        "baseline_checkpoint_ref": expected_baseline_checkpoint_ref,
        "candidate_checkpoint_ref": expected_candidate_checkpoint_ref,
    }
    if any(witness.get(key) != value for key, value in exact.items()):
        raise PermissionError("independent attestation is not scoped to these exact evidence bytes")
    if any(witness.get(key) is not True for key in (
        "holdout_not_disclosed_to_training",
        "holdout_labels_blinded_until_predictions_fixed",
        "predictions_observed_by_independent_evaluator",
        "checkpoint_bytes_verified_by_independent_evaluator",
    )):
        raise PermissionError("independent holdout custody or inference claims are missing")
    if _time(witness.get("holdout_sealed_at")) >= _time(witness.get("training_started_at")):
        raise PermissionError("independently attested holdout sealing did not precede training")

    pack = read_verified_json(artifacts, approved_pack_ref)
    if (pack.get("schema") != "scientific-state-holdout.v1"
            or pack.get("split") != "sealed_holdout"):
        raise ValueError("approved pack is not a sealed scientific holdout")
    cases = pack.get("cases")
    if not isinstance(cases, list) or len(cases) > 10000:
        raise ValueError("invalid holdout case count")
    sealed_ids = set()
    sealed_hashes = set()
    pair_members: dict[str, list[str]] = {}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("invalid sealed case")
        case_id = case.get("case_id")
        fp = case.get("example_fingerprint")
        if (not isinstance(case_id, str) or not case_id or
                not isinstance(fp, str) or len(fp) != 71 or
                not fp.startswith("sha256:") or
                any(x not in "0123456789abcdef" for x in fp[7:]) or
                case_id in sealed_ids or fp in sealed_hashes):
            raise ValueError("independent holdout requires unique hashed example content")
        sealed_ids.add(case_id)
        sealed_hashes.add(fp)
        pair_members.setdefault(case.get("contrast_id"), []).append(case_id)
    if any(len(x) != 2 for x in pair_members.values()):
        raise ValueError("unpaired independent contrast cases")

    for reference, schema in (
        (training_exposure_ref, "scientific-training-exposure.v1"),
        (development_exposure_ref, "scientific-development-exposure.v1"),
    ):
        other_ids, other_hashes = _fingerprints(
            read_verified_json(artifacts, reference), schema=schema)
        if sealed_ids & other_ids or sealed_hashes & other_hashes:
            raise PermissionError("sealed holdout overlaps training or development exposure")
    training_ids, training_hashes = _fingerprints(
        read_verified_json(artifacts, training_exposure_ref),
        schema="scientific-training-exposure.v1")
    development_ids, development_hashes = _fingerprints(
        read_verified_json(artifacts, development_exposure_ref),
        schema="scientific-development-exposure.v1")
    if training_ids & development_ids or training_hashes & development_hashes:
        raise PermissionError("training and development exposures overlap")

    assessment = assess_scientific_states(
        artifacts, approved_pack_ref=approved_pack_ref,
        baseline_refs=baseline_refs, candidate_refs=candidate_refs,
        diagnostic_manifest_ref=diagnostic_manifest_ref,
        minimum_accuracy=minimum_accuracy,
        minimum_contrast_accuracy=minimum_contrast_accuracy,
        minimum_per_state_accuracy=minimum_per_state_accuracy,
        minimum_matched_seeds=minimum_matched_seeds,
    )
    recorded = read_verified_json(artifacts, assessment.artifact_ref)
    if (recorded["baseline_checkpoint_ref"] != expected_baseline_checkpoint_ref
            or recorded["candidate_checkpoint_ref"] != expected_candidate_checkpoint_ref):
        raise PermissionError("scientific checkpoint differs from independent signed witness")

    # Each contrast pair is one statistical unit; repeated seeds are not
    # independently counted as new cases (avoids exaggerated significance).
    wins = losses = 0
    raw_baselines = [read_verified_json(artifacts, ref) for ref in baseline_refs]
    raw_candidates = [read_verified_json(artifacts, ref) for ref in candidate_refs]
    gold = {case["case_id"]: case["gold_state"] for case in cases}
    for members in pair_members.values():
        score_baseline = score_candidate = 0
        for baseline, candidate in zip(raw_baselines, raw_candidates, strict=True):
            bp = {row["case_id"]: row["state"] for row in baseline["predictions"]}
            cp = {row["case_id"]: row["state"] for row in candidate["predictions"]}
            score_baseline += all(bp[c] == gold[c] for c in members)
            score_candidate += all(cp[c] == gold[c] for c in members)
        if score_candidate > score_baseline:
            wins += 1
        elif score_candidate < score_baseline:
            losses += 1
    p = _exact_sign_test(wins, losses)
    reasons = list(assessment.blocking_issues)
    if assessment.candidate_accuracy - assessment.baseline_accuracy < minimum_practical_gain:
        reasons.append("practical_effect_below_preregistered_minimum")
    if wins + losses < minimum_decisive_contrast_pairs:
        reasons.append("insufficient_independent_decisive_contrast_pairs")
    if p > maximum_pair_p_value:
        reasons.append("paired_sign_test_not_significant")
    report = {
        "schema": "independent-evaluator-holdout-review.v1",
        "holdout_ref": approved_pack_ref,
        "assessment_ref": assessment.artifact_ref,
        "custody_signature_ref": attestation_ref,
        "trusted_custodian_id": trusted_custodian_id,
        "training_exposure_ref": training_exposure_ref,
        "development_exposure_ref": development_exposure_ref,
        "baseline_checkpoint_ref": expected_baseline_checkpoint_ref,
        "candidate_checkpoint_ref": expected_candidate_checkpoint_ref,
        "minimum_practical_gain": minimum_practical_gain,
        "minimum_decisive_contrast_pairs": minimum_decisive_contrast_pairs,
        "maximum_pair_p_value": maximum_pair_p_value,
        "observed_accuracy_gain": assessment.candidate_accuracy - assessment.baseline_accuracy,
        "winning_contrast_pairs": wins,
        "losing_contrast_pairs": losses,
        "decisive_pairs": wins + losses,
        "one_sided_pair_sign_test_p": p,
        "significance_assumption": "contrast_pairs_independently_sampled",
        "blocking_issues": reasons,
        "eligible_for_external_certification_review": not reasons,
        "certification_state": "not_certified",
        "live_activation_authorized": False,
        "caveat": "Signed witness verifies issuer identity, not truthfulness or absence of semantic leakage.",
    }
    receipt = artifacts.put_bytes(canonical_bytes(report), media_type="application/json")
    return IndependentHoldoutReview(
        assessment, receipt.artifact_ref, not reasons, tuple(reasons), wins, losses, p)


def verify_external_evaluator_certificate(
    artifacts: ArtifactStore, *, certificate_ref: str,
    expected_checkpoint_ref: str, pre_registered_pack_ref: str,
    trusted_certifier_id: str, trusted_certifier_public_key: bytes,
    trusted_custodian_id: str, trusted_custodian_public_key: bytes,
) -> dict:
    """Verify an external, separately signed certificate by re-running raw checks.

    The certifier's key and the custodian's key must be independently installed
    operator trust roots. This verifier neither issues nor applies certificates.
    """
    if (len(trusted_certifier_public_key) != 32
            or trusted_certifier_public_key == trusted_custodian_public_key
            or not trusted_certifier_id or trusted_certifier_id == trusted_custodian_id):
        raise PermissionError("scientific reviewer and holdout custodian must have separate trust roots")
    envelope = read_verified_json(artifacts, certificate_ref)
    if envelope.get("schema") != "independent-evaluator-certificate-signature.v1":
        raise PermissionError("independent certificate signature missing")
    statement_ref = envelope.get("statement_ref")
    statement = read_verified_json(artifacts, statement_ref)
    if (statement.get("schema") != "independent-evaluator-certificate.v1"
            or statement.get("certifier_id") != trusted_certifier_id
            or statement.get("role") != "evaluator"
            or statement.get("verdict") != "certified"
            or statement.get("candidate_checkpoint_ref") != expected_checkpoint_ref
            or statement.get("holdout_ref") != pre_registered_pack_ref):
        raise PermissionError("certificate is not authorized for this exact Evaluator evidence")
    try:
        signature = base64.b64decode(envelope["ed25519_signature_base64"], validate=True)
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        Ed25519PublicKey.from_public_bytes(trusted_certifier_public_key).verify(
            signature, canonical_bytes(statement))
    except (ValueError, KeyError, binascii.Error, InvalidSignature) as exc:
        raise PermissionError("independent certificate signature invalid") from exc
    if _time(statement.get("issued_at")) >= _time(statement.get("expires_at")):
        raise PermissionError("independent certificate time range invalid")
    if not _time(statement.get("issued_at")) <= datetime.now(timezone.utc) < _time(statement.get("expires_at")):
        raise PermissionError("independent certificate is not currently valid")

    report = read_verified_json(artifacts, statement.get("review_receipt_ref"))
    assessment = read_verified_json(artifacts, report.get("assessment_ref"))
    criteria = assessment.get("criteria", {})
    if (report.get("schema") != "independent-evaluator-holdout-review.v1"
            or not report.get("eligible_for_external_certification_review")
            or report.get("blocking_issues")
            or report.get("certification_state") != "not_certified"
            or report.get("live_activation_authorized") is not False
            or statement.get("holdout_ref") != report.get("holdout_ref")
            or report.get("candidate_checkpoint_ref") != expected_checkpoint_ref
            or assessment.get("candidate_checkpoint_ref") != expected_checkpoint_ref
            or statement.get("assessment_ref") != report.get("assessment_ref")
            or report.get("holdout_ref") != pre_registered_pack_ref):
        raise PermissionError("candidate is not backed by an independently review-eligible assessment")
    witness = _verify_signature(
        artifacts, attestation_ref=report["custody_signature_ref"],
        trusted_public_key=trusted_custodian_public_key,
        trusted_custodian_id=trusted_custodian_id)
    if witness.get("custodian_id") == statement.get("certifier_id"):
        raise PermissionError("holdout custodian cannot certify its own attestation")
    checked = review_independent_evaluator_holdout(
        artifacts, approved_pack_ref=pre_registered_pack_ref,
        pre_registered_pack_ref=pre_registered_pack_ref,
        diagnostic_manifest_ref=assessment["diagnostic_manifest_ref"],
        training_exposure_ref=report["training_exposure_ref"],
        development_exposure_ref=report["development_exposure_ref"],
        baseline_refs=assessment["baseline_refs"],
        candidate_refs=assessment["candidate_refs"],
        attestation_ref=report["custody_signature_ref"],
        trusted_custodian_id=trusted_custodian_id,
        trusted_custodian_public_key=trusted_custodian_public_key,
        expected_baseline_checkpoint_ref=report["baseline_checkpoint_ref"],
        expected_candidate_checkpoint_ref=expected_checkpoint_ref,
        minimum_accuracy=criteria["minimum_accuracy"],
        minimum_contrast_accuracy=criteria["minimum_contrast_accuracy"],
        minimum_per_state_accuracy=criteria["minimum_per_state_accuracy"],
        minimum_matched_seeds=criteria["minimum_matched_seeds"],
        minimum_practical_gain=report["minimum_practical_gain"],
        minimum_decisive_contrast_pairs=report["minimum_decisive_contrast_pairs"],
        maximum_pair_p_value=report["maximum_pair_p_value"],
    )
    if (not checked.eligible_for_external_certification_review
            or checked.review_receipt_ref != statement["review_receipt_ref"]
            or checked.assessment.artifact_ref != statement["assessment_ref"]):
        raise PermissionError("independent certificate fails recomputed scientific evidence")
    return {
        "certificate_ref": certificate_ref,
        "review_receipt_ref": checked.review_receipt_ref,
        "candidate_checkpoint_ref": expected_checkpoint_ref,
        "certifier_id": trusted_certifier_id,
        "status": "external_certificate_signature_verified",
        "caveat": "External signatures cannot independently verify custodian truthfulness.",
    }
