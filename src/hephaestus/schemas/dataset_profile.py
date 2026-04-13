from __future__ import annotations

from dataclasses import dataclass, field

from ._base import JsonSchema


@dataclass(slots=True)
class DatasetProfile(JsonSchema):
    dataset_id: str
    source_identity: str
    license: str
    quality_score: float
    risks: list[str] = field(default_factory=list)
