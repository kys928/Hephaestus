from __future__ import annotations

from hephaestus.schemas.incident_record import IncidentRecord
from hephaestus.schemas.runtime_event import RuntimeEvent, RuntimeEventCategory


def incident_from_event(event: RuntimeEvent) -> IncidentRecord | None:
    if event.category is not RuntimeEventCategory.INCIDENT:
        return None
    return IncidentRecord(
        incident_id=f"inc-{event.event_id}",
        run_id=event.run_id,
        severity="high" if "hard" in event.message else "medium",
        category="runtime",
        summary=event.message,
        event_ref=event.payload_ref,
    )


def launch_failure_incident(run_id: str, summary: str) -> IncidentRecord:
    return IncidentRecord(
        incident_id=f"inc-{run_id}-launch-failure",
        run_id=run_id,
        severity="high",
        category="runtime_launch",
        summary=summary,
        event_ref=None,
    )
