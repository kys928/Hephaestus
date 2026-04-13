from __future__ import annotations

from dataclasses import dataclass, field

from hephaestus.schemas.runtime_event import RuntimeEvent


@dataclass(slots=True)
class RuntimeEventStream:
    events: list[RuntimeEvent] = field(default_factory=list)

    def emit(self, event: RuntimeEvent) -> None:
        self.events.append(event)
