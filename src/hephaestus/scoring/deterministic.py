from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DeterministicCheckResult:
    name: str
    passed: bool
    details: str


def run_deterministic_checks(probe_score: float, toxicity: float, min_probe_score: float, max_toxicity: float) -> list[DeterministicCheckResult]:
    return [
        DeterministicCheckResult(
            name="probe_score_gate",
            passed=probe_score >= min_probe_score,
            details=f"probe_score={probe_score:.3f} min={min_probe_score:.3f}",
        ),
        DeterministicCheckResult(
            name="toxicity_gate",
            passed=toxicity <= max_toxicity,
            details=f"toxicity={toxicity:.3f} max={max_toxicity:.3f}",
        ),
    ]
