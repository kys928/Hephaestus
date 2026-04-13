from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RuntimePolicy:
    def classify(self, incident_count: int, deterministic_failures: int) -> str:
        if deterministic_failures > 0:
            return "hard_abort"
        if incident_count >= 3:
            return "waste_stop"
        if incident_count > 0:
            return "soft_suspicion"
        return "healthy"
