"""Schema: DatasetManifest."""

from __future__ import annotations

from dataclasses import dataclass

from ._base import JsonSchema


@dataclass(slots=True)
class DatasetManifest(JsonSchema):
    manifest_id: str
    datasets: list[str]
    total_examples: int
