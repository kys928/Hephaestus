from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ._base import JsonSchema


class RuntimeEventCategory(str, Enum):
    STATUS = "status"
    METRIC = "metric"
    PROBE = "probe"
    DETERMINISTIC_CHECK = "deterministic_check"
    INCIDENT = "incident"


@dataclass(slots=True)
class RuntimeEvent(JsonSchema):
    event_id: str
    run_id: str
    step: int
    category: RuntimeEventCategory
    message: str
    payload_ref: str | None = None
