from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DeterministicCheckResult:
    name: str
    passed: bool
    details: str


_CHECK_REGISTRY = {
    "probe_score_gate": ("probe_score", ">=", "min_probe_score"),
    "toxicity_gate": ("toxicity", "<=", "max_toxicity"),
}


def _check(metric_name: str, op: str, gate_name: str, metrics: dict[str, float], gates: dict[str, float], check_name: str) -> DeterministicCheckResult:
    value = float(metrics.get(metric_name, 0.0))
    target = float(gates[gate_name])
    passed = value >= target if op == ">=" else value <= target
    return DeterministicCheckResult(
        name=check_name,
        passed=passed,
        details=f"{metric_name}={value:.3f} {op} {target:.3f}",
    )


def run_deterministic_checks(metrics: dict[str, float], gates: dict[str, float], required_checks: list[str] | None = None) -> list[DeterministicCheckResult]:
    names = required_checks or list(_CHECK_REGISTRY.keys())
    results: list[DeterministicCheckResult] = []
    for name in names:
        if name not in _CHECK_REGISTRY:
            results.append(DeterministicCheckResult(name=name, passed=False, details="unknown_deterministic_check"))
            continue
        metric_name, op, gate_name = _CHECK_REGISTRY[name]
        if gate_name not in gates:
            results.append(DeterministicCheckResult(name=name, passed=False, details=f"missing_gate={gate_name}"))
            continue
        results.append(_check(metric_name, op, gate_name, metrics, gates, name))
    return results
