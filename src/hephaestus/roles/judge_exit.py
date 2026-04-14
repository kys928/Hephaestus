from __future__ import annotations

from dataclasses import dataclass

from hephaestus.policy.judge_policy import JudgePolicy
from hephaestus.policy.promotion_policy import PromotionPolicy
from hephaestus.schemas.eval_report import EvalReport
from hephaestus.schemas.judge_exit import JudgeExit, JudgeExitAction
from hephaestus.schemas.stage_profile import StageProfile


@dataclass(slots=True)
class JudgeExitRole:
    judge_policy: JudgePolicy
    promotion_policy: PromotionPolicy
    name: str = "judge_exit"

    def run(
        self,
        run_id: str,
        lineage_id: str,
        eval_report: EvalReport,
        monitor_outcome: str,
        recent_failure_count: int,
        has_stable_checkpoint: bool,
        stage_profile: StageProfile | None = None,
    ) -> JudgeExit:
        regression = eval_report.regression_summary
        deterministic_passed = bool(regression["deterministic_passed"])
        candidate_ref = str(eval_report.checkpoint_resolution.get("selected_checkpoint_ref", ""))
        has_candidate = bool(candidate_ref)
        promotion_state = self.promotion_policy.decide(
            deterministic_passed=deterministic_passed,
            confidence=eval_report.confidence,
            has_candidate=has_candidate,
        )
        action = self.judge_policy.decide_exit_action(
            deterministic_passed=deterministic_passed,
            confidence=eval_report.confidence,
            monitor_outcome=monitor_outcome,
            has_candidate_checkpoint=has_candidate,
            recent_failure_count=recent_failure_count,
            has_stable_checkpoint=has_stable_checkpoint,
        )
        if monitor_outcome == "healthy" and stage_profile and stage_profile.allowed_next_actions:
            allowed = set(stage_profile.allowed_next_actions)
            if action.value not in allowed:
                fallback = self._first_allowed(stage_profile.allowed_next_actions)
                if fallback is not None:
                    action = fallback
        verdict = "approved" if action.value in {"promote_checkpoint", "continue_from_checkpoint", "continue_lineage_best", "rollback_to_checkpoint"} else "blocked"
        return JudgeExit(
            run_id=run_id,
            lineage_id=lineage_id,
            verdict=verdict,
            next_action=action,
            confidence=eval_report.confidence,
            reasons=[
                f"monitor={monitor_outcome}",
                f"deterministic_passed={deterministic_passed}",
                f"promotion_state={promotion_state}",
            ],
        )

    def _first_allowed(self, actions: list[str]) -> JudgeExitAction | None:
        for value in actions:
            try:
                return JudgeExitAction(value)
            except ValueError:
                continue
        return None
