"""Verify externally provisioned trust and four frozen manifest identities.

Does not invent or sign custody; hash metadata cannot prove historical secrecy.
The caller must pin both the custodian public key and statement digest outside
candidate-controlled inputs. This verifier does not expose protected text.
"""
from __future__ import annotations

import base64
from datetime import datetime
import re

from hephaestus.schemas.custodial_references import CustodialReferenceManifest
from hephaestus.storage.verified import canonical_bytes, read_verified_json

ROLES = frozenset({"training", "development", "diagnostic", "holdout"})


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value)


def verify_custodial_reference_bundle(artifacts, *, signature_ref, trusted_statement_ref,
                                     trusted_public_key, trusted_custodian_id,
                                     expected_scope_ref, expected_reference_refs,
                                     now: datetime):
    if (not _digest(trusted_statement_ref) or not _digest(expected_scope_ref)
            or not trusted_custodian_id or len(trusted_public_key) != 32
            or set(expected_reference_refs) != ROLES
            or any(not _digest(r) for r in expected_reference_refs.values())
            or len(set(expected_reference_refs.values())) != 4):
        raise PermissionError("independently pinned custody roots and four distinct references required")
    try:
        scope = read_verified_json(artifacts, expected_scope_ref)
    except (ValueError, FileNotFoundError) as exc:
        raise PermissionError("independently accepted protected-audit scope is unavailable or corrupted") from exc
    hashes = scope.get("source_file_hashes")
    purposes = scope.get("purposes")
    if (scope.get("schema") != "custodial-contamination-audit-scope.v1"
            or scope.get("training_allowed") is not False
            or scope.get("protected_reference_access_allowed") is not True
            or scope.get("execution_boundary") != "custodian_environment"
            or scope.get("reference_manifest_refs") != expected_reference_refs
            or not isinstance(hashes, dict) or not hashes
            or any(not isinstance(k, str) or not k or not _digest(v) for k, v in hashes.items())
            or not isinstance(purposes, list) or any(not isinstance(p, str) for p in purposes)
            or not {"exact_near_duplicate", "semantic_contamination", "frozen_evaluation_leakage"}.issubset(purposes)):
        raise PermissionError("independently authorized protected-audit scope required; source-only permission is insufficient")
    envelope = read_verified_json(artifacts, signature_ref)
    if (envelope.get("schema") != "custodial-reference-signature.v1"
            or envelope.get("statement_ref") != trusted_statement_ref):
        raise PermissionError("custodial statement differs from independently accepted digest")
    statement = read_verified_json(artifacts, trusted_statement_ref)
    if (statement.get("schema") != "custodial-reference-access.v1"
            or statement.get("custodian_id") != trusted_custodian_id
            or statement.get("reference_manifest_refs") != expected_reference_refs
            or statement.get("scope_ref") != expected_scope_ref
            or statement.get("training_allowed") is not False
            or statement.get("execution_boundary") != "custodian_environment"
            or statement.get("original_text_audits_authorized") is not True):
        raise PermissionError("custody statement does not cover exact screening-only scope")
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        sig = base64.b64decode(envelope["signature_base64"], validate=True)
        Ed25519PublicKey.from_public_bytes(trusted_public_key).verify(sig, canonical_bytes(statement))
        issued = datetime.fromisoformat(statement["issued_at"].replace("Z", "+00:00"))
        expires = datetime.fromisoformat(statement["expires_at"].replace("Z", "+00:00"))
        if issued.utcoffset() is None or expires.utcoffset() is None or now.utcoffset() is None:
            raise ValueError("timezone required")
        if not issued <= now < expires:
            raise ValueError("not current")
    except Exception as exc:
        raise PermissionError("custodial signature or validity interval rejected") from exc
    manifests = {}
    for role, ref in expected_reference_refs.items():
        manifest = CustodialReferenceManifest(**read_verified_json(artifacts, ref))
        if (manifest.schema != "custodial-reference-manifest.v1" or manifest.role != role
                or manifest.custodian_id != trusted_custodian_id or not _digest(manifest.content_ref)
                or type(manifest.row_count) is not int or manifest.row_count <= 0
                or not manifest.normalization_profile
                or manifest.original_text_available_to_custodian is not True
                or manifest.exposure_inventory_complete is not True):
            raise PermissionError("incomplete or mismatched custodial reference manifest")
        try:
            sealed = datetime.fromisoformat(manifest.sealed_at.replace("Z", "+00:00"))
            if sealed.utcoffset() is None or sealed > issued:
                raise ValueError("sealing after authorization")
        except (ValueError, TypeError, AttributeError) as exc:
            raise PermissionError("invalid custodial sealing time") from exc
        manifests[role] = manifest.to_dict()
    if len({m["content_ref"] for m in manifests.values()}) != 4:
        raise PermissionError("reference contents must be distinct across roles")
    return manifests
