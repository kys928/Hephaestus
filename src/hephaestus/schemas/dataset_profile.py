"""Schema: DatasetProfile."""

from __future__ import annotations

from dataclasses import dataclass

from ._base import JsonSchema


@dataclass(slots=True)
class DatasetProfile(JsonSchema):
    dataset_id: str
    license: str
    source: str
