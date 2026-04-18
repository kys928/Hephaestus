from __future__ import annotations

from hephaestus.scoring.deterministic import run_deterministic_checks
from hephaestus.schemas.regression_summary import RegressionSummary


def build_regression_summary(
    run_id: str,
    metrics: dict[str, float],
    gates: dict[str, float],
    regression_bundles: dict[str, list[str]],
) -> RegressionSummary:
    checks = run_deterministic_checks(metrics=metrics, gates=gates)
    check_map = {check.name: check for check in checks}

    failed = [check.name for check in checks if not check.passed]
    notes = [check.details for check in checks]
    bundle_results: dict[str, dict[str, object]] = {}
    for bundle_name, bundle_checks in regression_bundles.items():
        missing_checks = [name for name in bundle_checks if name not in check_map]
        bundle_failed = [name for name in bundle_checks if name in check_map and not check_map[name].passed]
        bundle_results[bundle_name] = {
            "required_checks": bundle_checks,
            "passed": not bundle_failed and not missing_checks,
            "failed_checks": bundle_failed,
            "missing_checks": missing_checks,
        }

    return RegressionSummary(
        run_id=run_id,
        deterministic_passed=not failed,
        failed_checks=failed,
        notes=notes,
        bundle_results=bundle_results,
    )
