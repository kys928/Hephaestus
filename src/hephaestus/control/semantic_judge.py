"""Integration-owned Judge exit adapter for semantic experiment comparisons.

The semantic evaluator deliberately emits evidence and an advisory recommendation,
not an executable action. This adapter keeps that boundary explicit while routing
the comparison through the existing promotion and Judge policies.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from hephaestus.policy.judge_policy import JudgePolicy
from hephaestus.policy.promotion_policy import PromotionPolicy
from hephaestus.schemas.experiment_contract import ExperimentComparison
from hephaestus.schemas.judge_exit import JudgeExit, JudgeExitAction


_NON_CANDIDATE_OUTCOMES = {
    "equivalent_within_evidence",
    "mixed",
    "inconclusive",
    "invalid_comparison",
}
_HUMAN_REVIEW_RECOMMENDATIONS = {
    "consider_candidate_after_human_review",
    "human_review_required",
}


@dataclass(slots=True)
class SemanticComparisonJudgeAdapter:
    """Turn semantic comparison evidence into a governed finite Judge action.

    Only an ``improved`` comparison is allowed to enter the normal promotion
    policy. A semantic regression is explicitly rejected even when it did not
    trip a hard deterministic check. Equivalent, mixed, inconclusive, and
    invalid comparisons retain the baseline.

    Human-review requirements may only be satisfied by explicit caller evidence.
    When an independent review contributes a stronger governance confidence than
    the comparison's conservative no-judge ceiling, both confidences are retained
    in the reasons and the higher value is used only for promotion/certification
    policy. The semantic comparison's original confidence is never rewritten.
    """

    promotion_policy: PromotionPolicy = field(default_factory=PromotionPolicy)
    judge_policy: JudgePolicy = field(default_factory=JudgePolicy)

    def decide(
        self,
        comparison: ExperimentComparison,
        *,
        run_id: str,
        lineage_id: str,
        candidate_checkpoint_ref: str,
        monitor_outcome: str = "healthy",
        recent_failure_count: int = 0,
        has_stable_checkpoint: bool = False,
        human_review_approved: bool = False,
        human_review_approval_ref: str | None = None,
        independent_review_confidence: float | None = None,
        certification_requested: bool = False,
    ) -> JudgeExit:
        outcome = str(comparison.primary_outcome)
        deterministic_passed = comparison.deterministic_gate_status == "passed"
        has_candidate = bool(str(candidate_checkpoint_ref).strip())
        blocking_issues = [issue.code for issue in comparison.issues if issue.blocking]
        missing = comparison.effect_summary.get("missing_evidence", [])
        evidence_complete = not blocking_issues and not bool(missing)
        human_review_required = comparison.recommendation in _HUMAN_REVIEW_RECOMMENDATIONS
        human_review_pending = human_review_required and not human_review_approved

        review_confidence = None
        if independent_review_confidence is not None:
            review_confidence = max(0.0, min(1.0, float(independent_review_confidence)))
        governance_confidence = float(comparison.confidence)

        promotion_state = "inconclusive"
        certification_state = "certification_not_eligible"
        recheck_required = False
        observed_runs = 0
        direction_consistency = 0.0
        repeatability_sufficient = False

        if outcome == "regressed" or comparison.deterministic_gate_status == "failed":
            promotion_state = "rejected"
            certification_state = "certification_blocked_by_regression"
        elif outcome == "improved":
            repeatability = comparison.effect_summary.get("repeatability", {})
            if not isinstance(repeatability, dict):
                repeatability = {}
            observed_runs = max(1, int(repeatability.get("candidate_run_count", 1) or 1))
            direction_consistency = float(repeatability.get("direction_consistency", 0.0) or 0.0)
            repeatability_sufficient = bool(
                observed_runs >= 2
                and direction_consistency >= 0.67
                and comparison.variance_risk not in {"high", "unknown"}
            )

            # Independent review does not change the evaluator's semantic
            # confidence. It may raise governance confidence only after every
            # deterministic, completeness, repeatability, and variance condition
            # required for a promotion-capable bundle has already passed.
            review_can_contribute = bool(
                human_review_approved
                and review_confidence is not None
                and evidence_complete
                and deterministic_passed
                and repeatability_sufficient
                and comparison.variance_risk == "low"
            )
            if review_can_contribute:
                governance_confidence = max(governance_confidence, review_confidence)

            certification_ready = bool(
                certification_requested
                and review_can_contribute
                and observed_runs >= 3
                and review_confidence is not None
                and review_confidence >= self.promotion_policy.min_confidence_for_certified
            )
            requested_certification_state = (
                "certification_passed" if certification_ready else "certification_not_eligible"
            )

            promotion = self.promotion_policy.decide(
                deterministic_passed=deterministic_passed,
                confidence=governance_confidence,
                has_candidate=has_candidate,
                promotion_bundle_passed=evidence_complete and not human_review_pending,
                evidence_completeness=1.0 if evidence_complete else 0.0,
                certification_readiness=requested_certification_state,
                recheck_recommended=not repeatability_sufficient,
                observed_consistent_runs=observed_runs if direction_consistency >= 0.67 else 0,
                min_promotion_evidence=1,
                min_stable_evidence=2,
                observed_evidence_runs=observed_runs,
                min_certification_evidence=3,
                stability_confidence=governance_confidence,
                min_stability_confidence=0.9,
                repeatability_sufficient=repeatability_sufficient,
                variance_risk=comparison.variance_risk,
            )
            promotion_state = promotion.promotion_state
            certification_state = promotion.certification_state
            recheck_required = promotion.recheck_required

        action = self.judge_policy.decide_exit_action(
            deterministic_passed=deterministic_passed,
            confidence=governance_confidence,
            monitor_outcome=monitor_outcome,
            promotion_state=promotion_state,
            has_candidate_checkpoint=has_candidate,
            recent_failure_count=recent_failure_count,
            has_stable_checkpoint=has_stable_checkpoint,
        )

        # Semantic recommendations are evidence, not commands. These guards
        # ensure the finite Judge action cannot accidentally overtake the
        # evaluator's own stated evidence boundary.
        if outcome in _NON_CANDIDATE_OUTCOMES and action in {
            JudgeExitAction.PROMOTE_CHECKPOINT,
            JudgeExitAction.CONTINUE_FROM_CHECKPOINT,
        }:
            action = JudgeExitAction.CONTINUE_LINEAGE_BEST
        if human_review_pending and action == JudgeExitAction.PROMOTE_CHECKPOINT:
            action = JudgeExitAction.CONTINUE_FROM_CHECKPOINT

        approved_actions = {
            JudgeExitAction.PROMOTE_CHECKPOINT,
            JudgeExitAction.CONTINUE_FROM_CHECKPOINT,
            JudgeExitAction.CONTINUE_LINEAGE_BEST,
            JudgeExitAction.ROLLBACK_TO_CHECKPOINT,
        }
        return JudgeExit(
            run_id=run_id,
            lineage_id=lineage_id,
            verdict="approved" if action in approved_actions else "blocked",
            next_action=action,
            confidence=governance_confidence,
            reasons=[
                f"semantic_outcome={outcome}",
                f"semantic_recommendation={comparison.recommendation}",
                f"deterministic_gate_status={comparison.deterministic_gate_status}",
                f"promotion_state={promotion_state}",
                f"certification_state={certification_state}",
                f"human_review_required={human_review_required}",
                f"human_review_approved={human_review_approved}",
                f"human_review_approval_ref={human_review_approval_ref or 'none'}",
                f"human_review_pending={human_review_pending}",
                f"comparison_confidence={comparison.confidence}",
                f"independent_review_confidence={review_confidence if review_confidence is not None else 'none'}",
                f"governance_confidence={governance_confidence}",
                f"candidate_evidence_runs={observed_runs}",
                f"direction_consistency={direction_consistency}",
                f"repeatability_sufficient={repeatability_sufficient}",
                f"recheck_required={recheck_required}",
                f"blocking_issues={','.join(blocking_issues) if blocking_issues else 'none'}",
            ],
        )
