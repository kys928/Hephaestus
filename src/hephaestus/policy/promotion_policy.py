from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PromotionDecision:
    promotion_state: str
    certification_state: str
    recheck_required: bool
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PromotionPolicy:
    min_confidence_for_best: float = 0.6
    min_confidence_for_stable: float = 0.85
    min_confidence_for_certified: float = 0.93

    def decide(
        self,
        deterministic_passed: bool,
        confidence: float,
        has_candidate: bool = True,
        promotion_bundle_passed: bool = True,
        evidence_completeness: float = 1.0,
        certification_readiness: str = "certification_not_eligible",
        recheck_recommended: bool = False,
        stage_certification_eligibility: str = "standard",
        stage_require_recheck: bool = False,
        stage_min_consistent_runs: int = 1,
        observed_consistent_runs: int = 0,
        min_promotion_evidence: int = 1,
        min_stable_evidence: int = 1,
        observed_evidence_runs: int = 1,
        min_certification_evidence: int = 2,
        stability_confidence: float = 0.0,
        min_stability_confidence: float = 0.0,
        stage_thresholds: dict[str, float] | None = None,
    ) -> PromotionDecision:
        thresholds = stage_thresholds or {}
        best_threshold = float(thresholds.get("min_confidence_best", self.min_confidence_for_best))
        stable_threshold = float(thresholds.get("min_confidence_stable", self.min_confidence_for_stable))
        certified_threshold = float(thresholds.get("min_confidence_certified", self.min_confidence_for_certified))

        if not has_candidate:
            return PromotionDecision("inconclusive", "certification_not_eligible", recheck_recommended, ["missing_candidate_checkpoint"])

        if not deterministic_passed:
            return PromotionDecision("rejected", "certification_blocked_by_regression", recheck_recommended, ["deterministic_regression"])

        if confidence < best_threshold:
            return PromotionDecision("candidate_best", "certification_not_eligible", recheck_recommended, ["confidence_below_best_threshold"])
        if not promotion_bundle_passed:
            return PromotionDecision("candidate_best", "certification_blocked_by_regression", recheck_recommended, ["promotion_bundle_failed"])
        if observed_evidence_runs < min_promotion_evidence:
            return PromotionDecision("candidate_best", "certification_not_eligible", recheck_recommended, ["promotion_evidence_unmet"])

        promotion_state = "promoted_best"
        if (
            confidence >= stable_threshold
            and evidence_completeness >= 1.0
            and observed_evidence_runs >= min_stable_evidence
        ):
            promotion_state = "stable"

        if stage_certification_eligibility in {"disabled", "none", "ineligible"}:
            certification_state = "certification_not_eligible"
        else:
            certification_state = certification_readiness
        needs_recheck = recheck_recommended or (stage_require_recheck and observed_consistent_runs < stage_min_consistent_runs)
        if promotion_state == "stable" and certification_state == "certification_passed":
            if needs_recheck:
                certification_state = "certification_inconclusive"
            elif stability_confidence < min_stability_confidence:
                certification_state = "certification_inconclusive"
            elif confidence >= certified_threshold and observed_evidence_runs >= min_certification_evidence:
                promotion_state = "certified_stable"
                certification_state = "certification_passed"
            else:
                certification_state = "certification_inconclusive"

        return PromotionDecision(promotion_state, certification_state, recheck_recommended)
