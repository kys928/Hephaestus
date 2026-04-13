from __future__ import annotations

from dataclasses import dataclass

from hephaestus.backends.base import PreparedBackendJob
from hephaestus.runtime.event_stream import RuntimeEventStream
from hephaestus.runtime.launcher import LaunchResult


@dataclass(slots=True)
class RuntimeSession:
    run_id: str
    prepared_job: PreparedBackendJob
    launch_result: LaunchResult
    event_stream: RuntimeEventStream
    active: bool = True

    def close(self) -> None:
        self.active = False
