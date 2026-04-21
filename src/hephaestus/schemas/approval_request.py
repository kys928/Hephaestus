from __future__ import annotations

from dataclasses import dataclass, field

from ._base import JsonSchema


@dataclass(slots=True)
class ApprovalRequest(JsonSchema):
    request_id: str
    decision_id: str
    lineage_id: str
    run_id: str
    proposed_action: str
    reason: str
    risk_level: str
    required_approval_type: str
    status: str = "pending"
    created_at: str = ""
    metadata: dict[str, object] = field(default_factory=dict)
