from __future__ import annotations

from dataclasses import dataclass

from hephaestus.policy.promotion_policy import PromotionPolicy
from hephaestus.schemas.judge_entry import JudgeEntryMode
from hephaestus.schemas.judge_exit import JudgeExitAction


@dataclass(slots=True)
class JudgePolicy:
    promotion_policy: PromotionPolicy = PromotionPolicy()

    def decide_entry_mode(
        self,
        lineage_status: str,
        recent_failure_count: int,
        best_checkpoint_ref: str | None,
        last_stable_checkpoint_ref: str | None,
        parent_lineage_id: str | None,
    ) -> JudgeEntryMode:
        if lineage_status in {"poisoned", "restarted"}:
            return JudgeEntryMode.RESTART_LINEAGE
        if recent_failure_count >= 3 and last_stable_checkpoint_ref:
            return JudgeEntryMode.CONTINUE_FROM_LAST_STABLE
        if recent_failure_count >= 3:
            return JudgeEntryMode.RERUN_SAME_CONFIG
        if parent_lineage_id and best_checkpoint_ref:
            return JudgeEntryMode.BRANCH_FROM_CHECKPOINT
        if best_checkpoint_ref:
            return JudgeEntryMode.CONTINUE_LINEAGE_BEST
        return JudgeEntryMode.RERUN_SAME_CONFIG

    def decide_exit_action(
        self,
        deterministic_passed: bool,
        confidence: float,
        monitor_outcome: str,
        promotion_state: str | None = None,
        has_candidate_checkpoint: bool = True,
        recent_failure_count: int = 0,
        has_stable_checkpoint: bool = False,
    ) -> JudgeExitAction:
        if monitor_outcome == "hard_abort":
            return JudgeExitAction.ABORT_RUN
        if recent_failure_count >= 2 and has_stable_checkpoint:
            return JudgeExitAction.ROLLBACK_TO_CHECKPOINT
        if monitor_outcome == "waste_stop":
            return JudgeExitAction.RERUN_SAME_CONFIG
        if not has_candidate_checkpoint:
            return JudgeExitAction.CONTINUE_LINEAGE_BEST
        if promotion_state is None:
            promotion_state = self.promotion_policy.decide(
                deterministic_passed=deterministic_passed,
                confidence=confidence,
                has_candidate=has_candidate_checkpoint,
            ).promotion_state

        if promotion_state == "rejected":
            return JudgeExitAction.REJECT_CHECKPOINT
        if promotion_state in {"certified_stable", "stable", "promoted_best"}:
            return JudgeExitAction.PROMOTE_CHECKPOINT
        if promotion_state == "candidate_best":
            return JudgeExitAction.CONTINUE_FROM_CHECKPOINT
        return JudgeExitAction.CONTINUE_LINEAGE_BEST
