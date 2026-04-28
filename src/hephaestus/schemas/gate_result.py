from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._base import JsonSchema


@dataclass(slots=True)
class GateResult(JsonSchema):
    gate_id: str
    gate_name: str
    passed: bool
    severity: str
    reason: str
    blocking: bool
    evidence_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
