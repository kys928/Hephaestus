"""Schema: RegressionSummary."""

from __future__ import annotations

from dataclasses import dataclass

from ._base import JsonSchema


@dataclass(slots=True)
class RegressionSummary(JsonSchema):
    run_id: str
    checks_passed: bool
    notes: list[str]
