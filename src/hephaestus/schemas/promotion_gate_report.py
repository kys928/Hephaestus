from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._base import JsonSchema


@dataclass(slots=True)
class PromotionGateReport(JsonSchema):
    report_id: str
    run_id: str
    lineage_id: str
    candidate_checkpoint_ref: str | None
    requested_action: str
    recommended_effective_action: str
    gate_results: list[dict[str, object]] = field(default_factory=list)
    blocking_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    promotion_allowed: bool = False
    rollback_allowed: bool = False
    branch_allowed: bool = False
    restart_allowed: bool = False
    reject_allowed: bool = True
    confidence_ceiling: float = 1.0
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
