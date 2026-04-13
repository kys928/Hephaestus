from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ReporterRole:
    name: str = "reporter"

    def run(self, run_id: str, judge_action: str, monitor_outcome: str) -> dict[str, str]:
        return {
            "run_id": run_id,
            "summary": f"run={run_id} monitor={monitor_outcome} judge_action={judge_action}",
        }
