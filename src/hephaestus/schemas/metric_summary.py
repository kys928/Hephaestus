"""Schema: MetricSummary."""

from __future__ import annotations

from dataclasses import dataclass

from ._base import JsonSchema


@dataclass(slots=True)
class MetricSummary(JsonSchema):
    metric_name: str
    value: float
    threshold: float
