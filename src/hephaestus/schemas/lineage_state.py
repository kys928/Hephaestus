"""Schema: LineageState."""

from __future__ import annotations

from dataclasses import dataclass

from ._base import JsonSchema


@dataclass(slots=True)
class LineageState(JsonSchema):
    lineage_id: str
    parent_lineage_id: str | None
    status: str
    latest_run_id: str | None = None
