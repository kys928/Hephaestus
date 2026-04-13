from __future__ import annotations

from dataclasses import dataclass, field

from ._base import JsonSchema


@dataclass(slots=True)
class LineageState(JsonSchema):
    lineage_id: str
    parent_lineage_id: str | None
    status: str
    stage_name: str
    latest_run_id: str | None = None
    best_checkpoint_ref: str | None = None
    last_decision_id: str | None = None
    run_count: int = 0
    artifact_refs: list[str] = field(default_factory=list)
    updated_at: str = ""
