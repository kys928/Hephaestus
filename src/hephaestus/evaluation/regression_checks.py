from __future__ import annotations

from hephaestus.scoring.deterministic import run_deterministic_checks
from hephaestus.schemas.regression_summary import RegressionSummary


def build_regression_summary(run_id: str, probe_score: float, toxicity: float, min_probe_score: float, max_toxicity: float) -> RegressionSummary:
    checks = run_deterministic_checks(probe_score, toxicity, min_probe_score, max_toxicity)
    failed = [check.name for check in checks if not check.passed]
    notes = [check.details for check in checks]
    return RegressionSummary(
        run_id=run_id,
        deterministic_passed=not failed,
        failed_checks=failed,
        notes=notes,
    )
