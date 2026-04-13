from __future__ import annotations

from dataclasses import dataclass

from hephaestus.policy.promotion_policy import PromotionPolicy
from hephaestus.schemas.judge_exit import JudgeExitAction


@dataclass(slots=True)
class JudgePolicy:
    promotion_policy: PromotionPolicy = PromotionPolicy()

    def decide_exit_action(self, deterministic_passed: bool, confidence: float, monitor_outcome: str) -> JudgeExitAction:
        if monitor_outcome == "hard_abort":
            return JudgeExitAction.ABORT_RUN
        if monitor_outcome == "waste_stop":
            return JudgeExitAction.RERUN_SAME_CONFIG
        action = self.promotion_policy.decide(deterministic_passed, confidence)
        return JudgeExitAction(action)
