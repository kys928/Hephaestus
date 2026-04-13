from __future__ import annotations

from dataclasses import dataclass, field

from ._base import JsonSchema


@dataclass(slots=True)
class JudgeEntry(JsonSchema):
    run_id: str
    lineage_id: str
    objective: str
    constraints: list[str] = field(default_factory=list)
    approved: bool = True
    stage_name: str = ""
