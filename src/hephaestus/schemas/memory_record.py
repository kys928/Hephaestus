from __future__ import annotations

from dataclasses import dataclass, field

from ._base import JsonSchema


@dataclass(slots=True)
class MemoryRecord(JsonSchema):
    memory_id: str
    memory_type: str
    source_kind: str
    source_id: str
    lineage_id: str | None = None
    run_id: str | None = None
    stage_name: str | None = None
    created_at: str | None = None
    severity: str = "warning"
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    related_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)
