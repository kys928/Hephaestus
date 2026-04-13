from __future__ import annotations

from dataclasses import dataclass, field

from ._base import JsonSchema


@dataclass(slots=True)
class RunRecord(JsonSchema):
    run_id: str
    lineage_id: str
    stage_name: str
    status: str
    artifact_root: str
    started_at: str
    completed_at: str | None = None
    phase_order: list[str] = field(default_factory=list)
    monitor_outcome: str | None = None
    eval_report_id: str | None = None
    judge_action: str | None = None
