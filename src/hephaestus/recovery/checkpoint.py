"""Strict checkpoint-resume eligibility checks."""

from __future__ import annotations

from hephaestus.recovery.models import (
    CheckpointRecoveryDecision,
    NormalizedFailureEvidence,
    RecoveryRequest,
)
from hephaestus.schemas.contract_common import ContractIssue

_COMPATIBILITY_FIELDS = (
    "model_revision",
    "tokenizer_ref",
    "architecture_family",
    "training_recipe_ref",
    "data_contract_ref",
    "data_contract_hash",
    "backend_id",
)


def validate_checkpoint_recovery(
    request: RecoveryRequest,
    evidence: list[NormalizedFailureEvidence],
) -> CheckpointRecoveryDecision:
    checkpoint = _latest(evidence, {"checkpoint", "checkpoint_record"})
    token = _latest(evidence, {"resume_token", "resume_token_record"})
    replay = _latest(evidence, {"replay_verification", "replay_verification_report"})
    lineage = _latest(evidence, {"lineage", "lineage_state"})
    issues: list[ContractIssue] = []
    refs = sorted(
        {
            item.source_ref
            for item in (checkpoint, token, replay, lineage)
            if item is not None
        }
    )

    if checkpoint is None:
        issues.append(_missing("checkpoint_record", refs))
        checkpoint_payload: dict[str, object] = {}
    else:
        checkpoint_payload = checkpoint.payload
        exists = checkpoint_payload.get("exists", True) is True
        integrity = str(checkpoint_payload.get("integrity_level") or "")
        content_hash = str(checkpoint_payload.get("content_hash") or "")
        hash_verified = checkpoint_payload.get("hash_verified") is True or (
            integrity == "content_hash_verified" and bool(content_hash)
        )
        if not exists:
            issues.append(
                _integrity("checkpoint_missing", "Checkpoint does not exist.", refs)
            )
        if not content_hash:
            issues.append(_missing("checkpoint_content_hash", refs))
        if not hash_verified:
            issues.append(
                _integrity(
                    "checkpoint_hash_unverified",
                    "Checkpoint content hash is not explicitly verified.",
                    refs,
                )
            )

    if token is None:
        issues.append(_missing("resume_token", refs))
        token_payload: dict[str, object] = {}
    else:
        token_payload = token.payload
        if token_payload.get("exists", True) is not True:
            issues.append(_missing("resume_token", refs))
        if token_payload.get("valid") is not True:
            issues.append(
                _integrity(
                    "resume_token_invalid",
                    "Resume token validity is not explicitly verified.",
                    refs,
                )
            )

    replay_status = str((replay.payload if replay else {}).get("status") or "")
    if replay is None:
        issues.append(_missing("replay_verification", refs))
    elif replay_status != "reproducible":
        issues.append(
            ContractIssue(
                code="replay_policy_blocks_resume",
                category="policy_blocked",
                message="Checkpoint resume requires reproducible replay evidence.",
                retryable=True,
                blocking=True,
                evidence_refs=refs,
                metadata={"observed_status": replay_status or "missing"},
            )
        )

    lineage_status = str((lineage.payload if lineage else {}).get("status") or "")
    if lineage is None:
        issues.append(_missing("lineage_state", refs))
    elif lineage_status in {"poisoned", "deprecated", "archived", "blocked"}:
        issues.append(
            ContractIssue(
                code="lineage_policy_blocks_resume",
                category="policy_blocked",
                message=f"Lineage status {lineage_status!r} forbids continuation.",
                blocking=True,
                evidence_refs=refs,
            )
        )

    checkpoint_compatibility = _compatibility(checkpoint_payload)
    token_compatibility = _compatibility(token_payload)
    expected_raw = request.constraints.get("expected_compatibility", {})
    expected = (
        {str(key): str(value) for key, value in expected_raw.items()}
        if isinstance(expected_raw, dict)
        else {}
    )
    for field_name in _COMPATIBILITY_FIELDS:
        checkpoint_value = checkpoint_compatibility.get(field_name, "")
        token_value = token_compatibility.get(field_name, "")
        if not checkpoint_value or not token_value:
            issues.append(_missing(f"resume_compatibility.{field_name}", refs))
            continue
        if checkpoint_value != token_value:
            issues.append(_incompatible(field_name, refs))
            continue
        if field_name in expected and expected[field_name] != checkpoint_value:
            issues.append(_incompatible(field_name, refs))

    checkpoint_ref = (
        str(
            checkpoint_payload.get("checkpoint_ref")
            or token_payload.get("checkpoint_ref")
            or ""
        )
        or None
    )
    token_ref = (
        str(
            token_payload.get("resume_token_ref")
            or token_payload.get("token_ref")
            or (token.source_ref if token else "")
        )
        or None
    )
    return CheckpointRecoveryDecision(
        allowed=not issues,
        checkpoint_ref=checkpoint_ref,
        resume_token_ref=token_ref,
        evidence_refs=refs,
        issues=_deduplicate(issues),
        compatibility=checkpoint_compatibility,
    )


def _latest(
    evidence: list[NormalizedFailureEvidence], kinds: set[str]
) -> NormalizedFailureEvidence | None:
    matches = [item for item in evidence if item.evidence_kind in kinds]
    return matches[-1] if matches else None


def _compatibility(payload: dict[str, object]) -> dict[str, str]:
    raw = payload.get("resume_compatibility", payload.get("compatibility", {}))
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def _missing(name: str, refs: list[str]) -> ContractIssue:
    return ContractIssue(
        code=f"missing_{name.replace('.', '_')}",
        category="missing_evidence",
        message=f"Resume requires verified {name} evidence.",
        retryable=True,
        blocking=True,
        evidence_refs=refs,
    )


def _integrity(code: str, message: str, refs: list[str]) -> ContractIssue:
    return ContractIssue(
        code=code,
        category="artifact_integrity",
        message=message,
        blocking=True,
        evidence_refs=refs,
    )


def _incompatible(field_name: str, refs: list[str]) -> ContractIssue:
    return ContractIssue(
        code=f"resume_{field_name}_mismatch",
        category="incompatible_candidate",
        message=f"Resume {field_name} evidence does not match exactly.",
        blocking=True,
        evidence_refs=refs,
    )


def _deduplicate(issues: list[ContractIssue]) -> list[ContractIssue]:
    unique: dict[str, ContractIssue] = {}
    for issue in issues:
        unique[issue.code] = issue
    return [unique[key] for key in sorted(unique)]
