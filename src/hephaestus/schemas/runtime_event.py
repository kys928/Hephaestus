"""Schema: RuntimeEvent."""

from __future__ import annotations

from dataclasses import dataclass

from ._base import JsonSchema


@dataclass(slots=True)
class RuntimeEvent(JsonSchema):
    event_id: str
    run_id: str
    level: str
    message: str
