"""Small lineage transition helpers to keep policy rules out of orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hephaestus.control.promotion import PromotionTransition, apply_promotion
from hephaestus.policy.promotion_policy import PromotionPolicy


@dataclass(slots=True)
class LineageSignalUpdate:
    promotion: PromotionTransition
    failures: list[str]
    known_pathologies: list[str]
    trust_level: str


def compute_lineage_signals(
    prior_state: dict[str, Any],
    run_id: str,
    run_status: str,
    action: str,
    checkpoint_ref: str | None,
    deterministic_passed: bool,
    confidence: float,
    promotion_policy: PromotionPolicy,
) -> LineageSignalUpdate:
    promotion_state = promotion_policy.decide(
        deterministic_passed=deterministic_passed,
        confidence=confidence,
        has_candidate=bool(checkpoint_ref),
    )
    promotion = apply_promotion(
        lineage_state=prior_state,
        candidate_checkpoint_ref=checkpoint_ref,
        promotion_state=promotion_state,
        deterministic_passed=deterministic_passed,
        confidence=confidence,
        stable_confidence_threshold=promotion_policy.min_confidence_for_stable,
    )

    failures = list(prior_state.get("recent_failures", []))
    if run_status != "completed" or action in {"reject_checkpoint", "abort_run", "rollback_to_checkpoint"}:
        failures.append(run_id)
    failures = failures[-5:]

    pathologies = list(prior_state.get("known_pathologies", []))
    if action == "reject_checkpoint":
        pathologies.append("deterministic_regression")
    if action == "rollback_to_checkpoint":
        pathologies.append("rollback_triggered")
    pathologies = pathologies[-5:]

    trust = "low" if len(failures) >= 3 else promotion.trust_level
    return LineageSignalUpdate(
        promotion=promotion,
        failures=failures,
        known_pathologies=pathologies,
        trust_level=trust,
    )
