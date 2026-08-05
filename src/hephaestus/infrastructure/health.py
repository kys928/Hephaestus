"""Process-local liveness and filesystem readiness checks."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .observability import EventSink, NullEventSink, StructuredEvent


@dataclass(frozen=True, slots=True)
class HealthReport:
    live: bool
    ready: bool
    checks: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {"live": self.live, "ready": self.ready, "checks": dict(self.checks)}


@dataclass(slots=True)
class HealthService:
    state_root: Path
    artifact_root: Path
    event_sink: EventSink = field(default_factory=NullEventSink)

    @staticmethod
    def _check_root(path: Path) -> str:
        try:
            path.mkdir(parents=True, exist_ok=True)
            if not path.is_dir():
                return "not_a_directory"
            if not os.access(path, os.R_OK | os.W_OK | os.X_OK):
                return "not_read_write_accessible"
            return "ok"
        except OSError as exc:
            return f"error:{type(exc).__name__}"

    def check(self) -> HealthReport:
        checks = {
            "state_root": self._check_root(self.state_root),
            "artifact_root": self._check_root(self.artifact_root),
        }
        ready = all(value == "ok" for value in checks.values())
        report = HealthReport(live=True, ready=ready, checks=checks)
        self.event_sink.emit(
            StructuredEvent.create(
                "health.checked",
                "health",
                severity="info" if ready else "error",
                attributes={"live": report.live, "ready": report.ready, **checks},
            )
        )
        return report
