from __future__ import annotations

from dataclasses import dataclass, field

from hephaestus.schemas.runtime_event import RuntimeEvent, RuntimeEventCategory


@dataclass(slots=True)
class RuntimeEventStream:
    events: list[RuntimeEvent] = field(default_factory=list)

    def emit(self, event: RuntimeEvent) -> None:
        self.events.append(event)


def events_from_process_output(run_id: str, stdout: str, stderr: str) -> list[RuntimeEvent]:
    events: list[RuntimeEvent] = []
    for idx, line in enumerate(stdout.splitlines()):
        if not line.startswith("EVENT|"):
            continue
        _, category_raw, step_raw, message, payload_ref = (line.split("|", 4) + [""])[:5]
        try:
            category = RuntimeEventCategory(category_raw)
        except ValueError:
            category = RuntimeEventCategory.STATUS
        events.append(
            RuntimeEvent(
                event_id=f"{run_id}-stdout-{idx}",
                run_id=run_id,
                step=int(step_raw) if step_raw.isdigit() else 0,
                category=category,
                message=message,
                payload_ref=payload_ref or None,
            )
        )
    if stderr.strip():
        events.append(
            RuntimeEvent(
                event_id=f"{run_id}-stderr",
                run_id=run_id,
                step=0,
                category=RuntimeEventCategory.INCIDENT,
                message="backend stderr observed",
                payload_ref=None,
            )
        )
    return events
