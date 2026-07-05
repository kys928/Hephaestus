from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class AggregateScore:
    score: float
    weight_sum: float
    count: int
    missing_metrics: list[str]

    def to_dict(self) -> dict[str, object]:
        return {"score": self.score, "weight_sum": self.weight_sum, "count": self.count, "missing_metrics": list(self.missing_metrics)}


def weighted_mean(metrics: dict[str, float], weights: dict[str, float]) -> AggregateScore:
    total = 0.0
    weight_sum = 0.0
    missing: list[str] = []
    for name, weight in sorted(weights.items()):
        if name not in metrics:
            missing.append(name)
            continue
        total += float(metrics[name]) * float(weight)
        weight_sum += float(weight)
    return AggregateScore(score=(total / weight_sum if weight_sum else 0.0), weight_sum=weight_sum, count=len(weights) - len(missing), missing_metrics=missing)


def pass_rate(results: Iterable[bool]) -> float:
    values = list(results)
    return sum(1 for value in values if value) / len(values) if values else 0.0
