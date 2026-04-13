from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RuntimePolicy:
    def classify(self, incident_count: int, deterministic_failures: int, stop_sensitivity: str = "normal") -> str:
        if deterministic_failures > 0:
            return "hard_abort"

        waste_threshold = 3 if stop_sensitivity != "high" else 2
        suspicion_threshold = 0

        if incident_count >= waste_threshold:
            return "waste_stop"
        if incident_count > suspicion_threshold:
            return "soft_suspicion"
        return "healthy"
