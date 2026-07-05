"""Deterministic score aggregation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class AggregatedScore:
    score: float
    metric_count: int
    weights: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    missing_metrics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "metric_count": self.metric_count,
            "weights": dict(sorted(self.weights.items())),
            "metrics": dict(sorted(self.metrics.items())),
            "missing_metrics": list(self.missing_metrics),
        }


def weighted_average(metrics: Mapping[str, float], weights: Mapping[str, float] | None = None, precision: int = 4) -> AggregatedScore:
    """Compute a stable weighted average over numeric metrics."""

    normalized_metrics = {str(name): float(value) for name, value in metrics.items()}
    if weights is None:
        normalized_weights = {name: 1.0 for name in normalized_metrics}
    else:
        normalized_weights = {str(name): float(value) for name, value in weights.items() if float(value) > 0.0}

    present = {name: normalized_metrics[name] for name in sorted(normalized_metrics) if name in normalized_weights}
    missing = sorted(name for name in normalized_weights if name not in normalized_metrics)
    total_weight = sum(normalized_weights[name] for name in present)
    score = 0.0 if total_weight <= 0.0 else sum(present[name] * normalized_weights[name] for name in present) / total_weight
    return AggregatedScore(round(score, precision), len(present), {name: normalized_weights[name] for name in sorted(normalized_weights)}, present, missing)


def aggregate_gate_results(gate_results: Mapping[str, Mapping[str, object]]) -> dict[str, Any]:
    """Summarize deterministic gate results into JSON-serializable lists."""

    passed: list[str] = []
    failed: list[str] = []
    for name in sorted(gate_results):
        if bool(gate_results[name].get("passed", False)):
            passed.append(str(name))
        else:
            failed.append(str(name))
    return {"deterministic_passed": not failed, "passed_gates": passed, "failed_gates": failed, "gate_count": len(passed) + len(failed)}


def aggregate_metrics(metrics: Mapping[str, float], weights: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Return the JSON dictionary form of ``weighted_average``."""

    return weighted_average(metrics, weights).to_dict()
