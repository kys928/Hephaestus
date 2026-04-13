from __future__ import annotations

from dataclasses import dataclass, field

from ._base import JsonSchema


@dataclass(slots=True)
class RegressionSummary(JsonSchema):
    run_id: str
    deterministic_passed: bool
    failed_checks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
