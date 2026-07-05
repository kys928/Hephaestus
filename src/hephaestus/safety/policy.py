"""Deterministic aggregation policy for safety guard outputs."""

from __future__ import annotations

from hephaestus.schemas.safety_guard import SafetyGuardResult, SafetyPolicyDecision


def decide_boundary(
    *,
    run_id: str,
    lineage_id: str,
    boundary: str,
    guard_results: list[SafetyGuardResult],
    fallback_action: str | None = None,
) -> SafetyPolicyDecision:
    ordered = sorted(guard_results, key=lambda item: item.guard_id)
    blocking = [item.guard_id for item in ordered if not item.allowed]
    warnings: list[str] = []
    for item in ordered:
        warnings.extend(f"{item.guard_id}:{warning}" for warning in item.warnings)
    return SafetyPolicyDecision(
        run_id=run_id,
        lineage_id=lineage_id,
        boundary=boundary,
        allowed=not blocking,
        effective_action=None if not blocking else fallback_action,
        blocking_guards=blocking,
        guard_results=[item.to_dict() for item in ordered],
        warnings=sorted(set(warnings)),
    )


def append_guard_metadata(target: dict[str, object], decision: SafetyPolicyDecision, key: str = "safety") -> None:
    target[key] = decision.to_dict()
