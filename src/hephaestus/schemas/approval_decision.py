from __future__ import annotations

from dataclasses import dataclass, field

from ._base import JsonSchema


@dataclass(slots=True)
class ApprovalDecision(JsonSchema):
    decision_event_id: str
    request_id: str
    lineage_id: str
    run_id: str
    operator_id: str
    outcome: str
    status: str
    note: str
    effect_on_action: str
    created_at: str = ""
    metadata: dict[str, object] = field(default_factory=dict)
