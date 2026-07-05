from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ._base import JsonSchema

ALLOWED_OPERATOR_ACTIONS = {"approve_code_edit", "reject_code_edit", "note"}


def _text(value: object | None, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


@dataclass(slots=True)
class OperatorAction(JsonSchema):
    action_id: str
    action_type: str
    requested_by: str = "unknown"
    target_type: str = ""
    target_id: str = ""
    status: str = "recorded"
    reason: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OperatorAction":
        return cls(
            action_id=_text(payload.get("action_id"), "operator-action-unknown"),
            action_type=_text(payload.get("action_type"), "note"),
            requested_by=_text(payload.get("requested_by"), "unknown"),
            target_type=_text(payload.get("target_type")),
            target_id=_text(payload.get("target_id")),
            status=_text(payload.get("status"), "recorded"),
            reason=_text(payload.get("reason")),
            created_at=_text(payload.get("created_at"), datetime.now(timezone.utc).isoformat()),
            metadata=dict(payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}),
        )
