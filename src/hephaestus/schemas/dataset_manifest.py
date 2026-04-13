from __future__ import annotations

from dataclasses import dataclass, field

from ._base import JsonSchema


@dataclass(slots=True)
class DatasetManifest(JsonSchema):
    manifest_id: str
    run_id: str
    lineage_id: str
    source_ids: list[str] = field(default_factory=list)
    total_examples: int = 0
    artifact_ref: str = ""
