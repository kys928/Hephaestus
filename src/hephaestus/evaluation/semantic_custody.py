"""Fail-closed admission of an independently conducted semantic leakage audit.

The training process cannot self-certify lexical or semantic isolation.
This verifier authenticates *scope and completeness of the supplied audit
receipt*, not the truth of its off-box similarity measurements. Its exact
content-addressed ref is included in the custodian's signed attestation.
"""
from __future__ import annotations

import hashlib
from datetime import datetime

from hephaestus.storage.base import ArtifactStore
from hephaestus.storage.verified import canonical_bytes, read_verified_json


def verify_semantic_leakage_audit(
    artifacts: ArtifactStore, *, audit_ref: str, holdout_ref: str,
    training_exposure_ref: str, development_exposure_ref: str,
    diagnostic_manifest_ref: str, holdout_case_ids: set[str],
    training_count: int, development_count: int, diagnostic_count: int,
    holdout_sealed_at: datetime,
) -> dict[str, object]:
    """Reject missing, unscoped, incomplete or positive-overlap audit receipts.

    The caller MUST authenticate audit_ref inside a separately signed external
    custody statement, with a trust root not controlled by the training agent.
    The auditor still has to examine private original examples outside this
    application; hashes alone cannot reveal paraphrases.
    """
    if not isinstance(audit_ref, str) or not audit_ref.startswith("sha256:"):
        raise PermissionError("independent semantic leakage audit is mandatory")
    audit = read_verified_json(artifacts, audit_ref)
    if audit.get("schema") != "independent-semantic-leakage-audit.v1":
        raise PermissionError("independent semantic leakage audit schema mismatch")
    expected = {
        "holdout_ref": holdout_ref,
        "training_exposure_ref": training_exposure_ref,
        "development_exposure_ref": development_exposure_ref,
        "diagnostic_manifest_ref": diagnostic_manifest_ref,
        "case_ids_sha256": "sha256:" + hashlib.sha256(
            canonical_bytes(sorted(holdout_case_ids))).hexdigest(),
        "holdout_cases_checked": len(holdout_case_ids),
        "training_cases_checked": training_count,
        "development_cases_checked": development_count,
        "diagnostic_cases_checked": diagnostic_count,
    }
    if any(audit.get(key) != value for key, value in expected.items()):
        raise PermissionError("semantic leakage audit does not cover exact sealed data and exposures")
    for field in ("holdout_cases_checked", "training_cases_checked",
                  "development_cases_checked", "diagnostic_cases_checked"):
        if type(audit.get(field)) is not int:
            raise PermissionError("semantic leakage audit has non-integer coverage")
    methods = audit.get("methods")
    if (not isinstance(methods, list) or not all(isinstance(v, str) for v in methods)
            or len(methods) != len(set(methods))
            or not {"lexical_near_duplicate", "semantic_similarity", "human_review"}.issubset(set(methods))):
        raise PermissionError("semantic leakage audit lacks lexical, semantic, or human review")
    if (not isinstance(audit.get("auditor_id"), str) or not audit["auditor_id"].strip()
            or not isinstance(audit.get("method_version"), str) or not audit["method_version"].strip()
            or audit.get("original_text_examined") is not True
            or audit.get("candidate_scores_blinded") is not True):
        raise PermissionError("semantic leakage audit provenance or blinding is incomplete")
    try:
        scan_time = datetime.fromisoformat(str(audit["scan_completed_at"]).replace("Z", "+00:00"))
    except (KeyError, ValueError) as exc:
        raise PermissionError("semantic leakage audit completion time is missing") from exc
    if scan_time.utcoffset() is None or scan_time < holdout_sealed_at:
        raise PermissionError("semantic leakage audit was not performed after holdout sealing")
    # A reported unresolved match is a failed gate, never a warning to ignore.
    if (audit.get("disposition") != "no_detected_leakage"
            or type(audit.get("suspected_near_duplicates")) is not int
            or audit["suspected_near_duplicates"] != 0
            or type(audit.get("unresolved_semantic_matches")) is not int
            or audit["unresolved_semantic_matches"] != 0):
        raise PermissionError("independent semantic leakage audit is inconclusive or found overlap")
    return audit
