from __future__ import annotations

from dataclasses import dataclass

from hephaestus.backends.dry_run_backend import DryRunBackend
from hephaestus.policy.runtime_policy import RuntimePolicy
from hephaestus.runtime.health_checks import count_deterministic_failures, count_incidents
from hephaestus.runtime.incident_manager import incident_from_event
from hephaestus.runtime.stop_logic import stop_recommendation
from hephaestus.schemas.incident_record import IncidentRecord
from hephaestus.schemas.runtime_event import RuntimeEvent


@dataclass(slots=True)
class RuntimeMonitorResult:
    outcome: str
    recommendation: str
    events: list[RuntimeEvent]
    incidents: list[IncidentRecord]


@dataclass(slots=True)
class RuntimeMonitorRole:
    backend: DryRunBackend
    runtime_policy: RuntimePolicy
    name: str = "runtime_monitor"

    def run(self, run_id: str) -> RuntimeMonitorResult:
        events = self.backend.runtime_events(run_id)
        incidents = [incident for event in events if (incident := incident_from_event(event))]
        outcome = self.runtime_policy.classify(
            incident_count=count_incidents(events),
            deterministic_failures=count_deterministic_failures(events),
        )
        return RuntimeMonitorResult(
            outcome=outcome,
            recommendation=stop_recommendation(outcome),
            events=events,
            incidents=incidents,
        )
