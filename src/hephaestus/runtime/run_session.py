from __future__ import annotations

from dataclasses import dataclass

from hephaestus.runtime.event_stream import RuntimeEventStream


@dataclass(slots=True)
class RuntimeSession:
    run_id: str
    event_stream: RuntimeEventStream
    active: bool = True

    def close(self) -> None:
        self.active = False
