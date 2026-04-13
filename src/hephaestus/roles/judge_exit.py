from __future__ import annotations

from dataclasses import dataclass

from hephaestus.policy.judge_policy import JudgePolicy
from hephaestus.schemas.eval_report import EvalReport
from hephaestus.schemas.judge_exit import JudgeExit, JudgeExitAction
from hephaestus.schemas.stage_profile import StageProfile


@dataclass(slots=True)
class JudgeExitRole:
    judge_policy: JudgePolicy
    name: str = "judge_exit"

    def run(
        self,
        run_id: str,
        lineage_id: str,
        eval_report: EvalReport,
        monitor_outcome: str,
        stage_profile: StageProfile | None = None,
    ) -> JudgeExit:
        regression = eval_report.regression_summary
        deterministic_passed = bool(regression["deterministic_passed"])
        action = self.judge_policy.decide_exit_action(
            deterministic_passed=deterministic_passed,
            confidence=eval_report.confidence,
            monitor_outcome=monitor_outcome,
        )
        if monitor_outcome == "healthy" and stage_profile and stage_profile.allowed_next_actions:
            allowed = set(stage_profile.allowed_next_actions)
            if action.value not in allowed:
                fallback = self._first_allowed(stage_profile.allowed_next_actions)
                if fallback is not None:
                    action = fallback
        verdict = "approved" if action.value in {"promote_checkpoint", "continue_from_checkpoint", "continue_lineage_best"} else "blocked"
        return JudgeExit(
            run_id=run_id,
            lineage_id=lineage_id,
            verdict=verdict,
            next_action=action,
            confidence=eval_report.confidence,
            reasons=[f"monitor={monitor_outcome}", f"deterministic_passed={deterministic_passed}"],
        )

    def _first_allowed(self, actions: list[str]) -> JudgeExitAction | None:
        for value in actions:
            try:
                return JudgeExitAction(value)
            except ValueError:
                continue
        return None
