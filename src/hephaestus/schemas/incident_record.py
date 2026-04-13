"""Schema: IncidentRecord."""

from __future__ import annotations

from dataclasses import dataclass

from ._base import JsonSchema


@dataclass(slots=True)
class IncidentRecord(JsonSchema):
    incident_id: str
    run_id: str
    severity: str
    summary: str
