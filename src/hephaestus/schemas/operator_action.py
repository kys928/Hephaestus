from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ._base import JsonSchema


OperatorActionKind = Literal["approval_decision", "replay_verification_request", "run_command"]


@dataclass(slots=True)
class OperatorActionRequest(JsonSchema):
    action: str
    operator_id: str
    run_id: str | None = None
    lineage_id: str | None = None
    request_id: str | None = None
    outcome: str | None = None
    note: str = ""
    reason: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class OperatorActionRecord(JsonSchema):
    action_event_id: str
    action_kind: OperatorActionKind
    action: str
    operator_id: str
    status: str
    created_at: str
    run_id: str | None = None
    lineage_id: str | None = None
    request_id: str | None = None
    note: str = ""
    reason: str = ""
    policy_decision: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
