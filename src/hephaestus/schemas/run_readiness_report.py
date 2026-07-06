from __future__ import annotations

from dataclasses import dataclass, field

from ._base import JsonSchema


@dataclass(slots=True)
class RunReadinessReport(JsonSchema):
    report_id: str
    run_id: str
    lineage_id: str
    stage_name: str
    stage_contract_id: str
    status: str
    launch_allowed: bool
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, dict[str, object]] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
