from __future__ import annotations

from hephaestus.schemas.safety_guard import SafetyGuardResult


def evaluate_launch_guard(launch_config: dict[str, object]) -> SafetyGuardResult:
    reasons: list[str] = []
    if not bool(launch_config.get("dry_run", True)) and not launch_config.get("approval_request_id"):
        reasons.append("non_dry_run_launch_requires_approval")
    return SafetyGuardResult(
        guard_id="launch_guard",
        passed=not reasons,
        severity="error" if reasons else "info",
        reasons=reasons,
        metadata={"dry_run": bool(launch_config.get("dry_run", True))},
    )
