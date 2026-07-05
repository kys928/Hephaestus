from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ._base import JsonSchema

CANONICAL_RISK_LEVELS = {"low", "medium", "high", "forbidden"}
CANONICAL_STATUSES = {
    "proposed",
    "approval_required",
    "approved",
    "rejected",
    "blocked",
    "executed",
    "superseded",
}


def _as_str(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_object_dict(value: object | None) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(k): v for k, v in value.items()}


def _as_str_list(value: object | None) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = _as_str(item)
        if text:
            out.append(text)
    return out


def normalize_code_edit_proposal(payload: dict[str, object]) -> dict[str, object]:
    proposed_id = _as_str(payload.get("proposal_id")) or "proposal-unknown"
    status = _as_str(payload.get("status")) or "proposed"
    if status not in CANONICAL_STATUSES:
        status = "proposed"

    risk_level = _as_str(payload.get("risk_level")) or "medium"
    if risk_level not in CANONICAL_RISK_LEVELS:
        risk_level = "medium"

    target_files = sorted(set(_as_str_list(payload.get("target_files"))))
    forbidden_files_touched = sorted(set(_as_str_list(payload.get("forbidden_files_touched"))))
    allowed_files_touched = sorted(set(_as_str_list(payload.get("allowed_files_touched"))))
    required_approvals = sorted(set(_as_str_list(payload.get("required_approvals"))))

    if "operator_approval" not in required_approvals:
        required_approvals.append("operator_approval")

    purpose = _as_str(payload.get("purpose"))
    rollback_plan = _as_str(payload.get("rollback_plan"))
    metadata = _as_object_dict(payload.get("metadata"))
    missing_fields = metadata.get("missing_required_fields")
    missing_required_fields: list[str] = (
        [str(v) for v in missing_fields if isinstance(v, str)]
        if isinstance(missing_fields, list)
        else []
    )

    if status != "blocked":
        if not purpose:
            purpose = "TODO: specify purpose"
            missing_required_fields.append("purpose")
        if not rollback_plan:
            rollback_plan = "TODO: specify rollback plan"
            missing_required_fields.append("rollback_plan")
    if missing_required_fields:
        metadata["missing_required_fields"] = sorted(set(missing_required_fields))

    if status not in {"rejected", "blocked", "approved", "executed", "superseded"}:
        status = "approval_required"

    return {
        "proposal_id": proposed_id,
        "run_id": _as_str(payload.get("run_id")),
        "lineage_id": _as_str(payload.get("lineage_id")),
        "created_at": _as_str(payload.get("created_at")) or datetime.now(timezone.utc).isoformat(),
        "requested_by": _as_str(payload.get("requested_by")) or "unknown",
        "purpose": purpose or "",
        "risk_level": risk_level,
        "status": status,
        "target_files": target_files,
        "forbidden_files_touched": forbidden_files_touched,
        "allowed_files_touched": allowed_files_touched,
        "patch_summary": _as_str(payload.get("patch_summary")) or "",
        "proposed_diff_ref": _as_str(payload.get("proposed_diff_ref")),
        "test_plan": _as_str_list(payload.get("test_plan")),
        "required_approvals": sorted(set(required_approvals)),
        "approval_request_id": _as_str(payload.get("approval_request_id")),
        "evidence_refs": _as_str_list(payload.get("evidence_refs")),
        "rollback_plan": rollback_plan,
        "metadata": metadata,
    }


@dataclass(slots=True)
class CodeEditProposal(JsonSchema):
    proposal_id: str
    run_id: str | None = None
    lineage_id: str | None = None
    created_at: str | None = None
    requested_by: str = "unknown"
    purpose: str = ""
    risk_level: str = "medium"
    status: str = "proposed"
    target_files: list[str] = field(default_factory=list)
    forbidden_files_touched: list[str] = field(default_factory=list)
    allowed_files_touched: list[str] = field(default_factory=list)
    patch_summary: str = ""
    proposed_diff_ref: str | None = None
    test_plan: list[str] = field(default_factory=list)
    required_approvals: list[str] = field(default_factory=list)
    approval_request_id: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    rollback_plan: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CodeEditProposal":
        return cls(**normalize_code_edit_proposal(payload))


@dataclass(slots=True)
class CodeEditExecutionRecord(JsonSchema):
    execution_id: str
    proposal_id: str
    run_id: str | None = None
    lineage_id: str | None = None
    requested_by: str = "unknown"
    status: str = "refused"
    dry_run: bool = True
    reason: str = ""
    target_files: list[str] = field(default_factory=list)
    created_at: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CodeEditExecutionRecord":
        return cls(
            execution_id=_as_str(payload.get("execution_id")) or "exec-unknown",
            proposal_id=_as_str(payload.get("proposal_id")) or "proposal-unknown",
            run_id=_as_str(payload.get("run_id")),
            lineage_id=_as_str(payload.get("lineage_id")),
            requested_by=_as_str(payload.get("requested_by")) or "unknown",
            status=_as_str(payload.get("status")) or "refused",
            dry_run=bool(payload.get("dry_run", True)),
            reason=_as_str(payload.get("reason")) or "",
            target_files=_as_str_list(payload.get("target_files")),
            created_at=_as_str(payload.get("created_at")) or datetime.now(timezone.utc).isoformat(),
            metadata=_as_object_dict(payload.get("metadata")),
        )
