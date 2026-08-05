"""Small-sample aggregation helpers with no significance overclaiming."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True, slots=True)
class SpreadSummary:
    count: int
    mean: float
    minimum: float
    maximum: float
    spread: float
    population_stddev: float

    def to_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "mean": self.mean,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "spread": self.spread,
            "population_stddev": self.population_stddev,
        }


def summarize_spread(values: Iterable[float]) -> SpreadSummary:
    observed = [float(value) for value in values]
    if not observed:
        return SpreadSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    minimum = min(observed)
    maximum = max(observed)
    return SpreadSummary(
        count=len(observed),
        mean=sum(observed) / len(observed),
        minimum=minimum,
        maximum=maximum,
        spread=maximum - minimum,
        population_stddev=statistics.pstdev(observed) if len(observed) > 1 else 0.0,
    )


def aggregate_dimensions(rows: Iterable[Mapping[str, float]]) -> dict[str, SpreadSummary]:
    values: dict[str, list[float]] = {}
    for row in rows:
        for dimension, score in row.items():
            values.setdefault(str(dimension), []).append(float(score))
    return {dimension: summarize_spread(scores) for dimension, scores in sorted(values.items())}


def practical_effect(delta: float, minimum_practical_improvement: float) -> str:
    threshold = abs(float(minimum_practical_improvement))
    if delta >= threshold:
        return "improved"
    if delta <= -threshold:
        return "regressed"
    return "equivalent_within_evidence"


def variance_risk(summary: SpreadSummary, moderate_threshold: float, high_threshold: float) -> str:
    if summary.count < 2:
        return "unknown"
    signal = max(summary.spread, summary.population_stddev * 2.0)
    if signal > high_threshold:
        return "high"
    if signal > moderate_threshold:
        return "moderate"
    return "low"


def bounded_repeatability_factor(observed: int, required: int) -> float:
    if observed <= 0:
        return 0.0
    target = max(1, int(required))
    return min(1.0, math.sqrt(observed / target))
