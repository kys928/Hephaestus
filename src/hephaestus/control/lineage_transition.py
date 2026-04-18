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
    promotion_bundle_passed: bool,
    evidence_completeness: float,
    certification_readiness: str,
    recheck_recommended: bool,
    stage_certification_eligibility: str,
    stage_require_recheck: bool,
    stage_min_consistent_runs: int,
    observed_consistent_runs: int,
    min_promotion_evidence: int,
    observed_evidence_runs: int,
    min_stable_evidence: int,
    min_certification_evidence: int,
    stability_confidence: float,
    min_stability_confidence: float,
    stage_thresholds: dict[str, float],
    promotion_policy: PromotionPolicy,
    repeatability_sufficient: bool,
    variance_risk: str,
) -> LineageSignalUpdate:
    decision = promotion_policy.decide(
        deterministic_passed=deterministic_passed,
        confidence=confidence,
        has_candidate=bool(checkpoint_ref),
        promotion_bundle_passed=promotion_bundle_passed,
        evidence_completeness=evidence_completeness,
        certification_readiness=certification_readiness,
        recheck_recommended=recheck_recommended,
        stage_certification_eligibility=stage_certification_eligibility,
        stage_require_recheck=stage_require_recheck,
        stage_min_consistent_runs=stage_min_consistent_runs,
        observed_consistent_runs=observed_consistent_runs,
        min_promotion_evidence=min_promotion_evidence,
        observed_evidence_runs=observed_evidence_runs,
        min_stable_evidence=min_stable_evidence,
        min_certification_evidence=min_certification_evidence,
        stability_confidence=stability_confidence,
        min_stability_confidence=min_stability_confidence,
        stage_thresholds=stage_thresholds,
        repeatability_sufficient=repeatability_sufficient,
        variance_risk=variance_risk,
    )
    promotion = apply_promotion(
        lineage_state=prior_state,
        candidate_checkpoint_ref=checkpoint_ref,
        promotion_state=decision.promotion_state,
        certification_state=decision.certification_state,
        deterministic_passed=deterministic_passed,
        confidence=confidence,
        stable_confidence_threshold=float(stage_thresholds.get("min_confidence_stable", promotion_policy.min_confidence_for_stable)),
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
    if decision.certification_state == "certification_inconclusive":
        pathologies.append("certification_inconclusive")
    if decision.certification_state == "certification_inconclusive_due_to_variance":
        pathologies.append("variance_blocked_certification")
    if decision.certification_state == "certification_blocked_by_inconsistency":
        pathologies.append("repeatability_inconsistent")
    if decision.certification_state == "certification_recheck_required":
        pathologies.append("certification_recheck_required")
    pathologies = pathologies[-5:]

    trust = "low" if len(failures) >= 3 else promotion.trust_level
    return LineageSignalUpdate(
        promotion=promotion,
        failures=failures,
        known_pathologies=pathologies,
        trust_level=trust,
    )
