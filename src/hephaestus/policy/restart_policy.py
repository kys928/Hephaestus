from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RestartPolicy:
    def decide(self, judge_action: str, monitor_outcome: str) -> str:
        if judge_action == "restart_lineage" or monitor_outcome == "hard_abort":
            return "restart"
        if monitor_outcome == "waste_stop":
            return "rollback"
        return "continue"
