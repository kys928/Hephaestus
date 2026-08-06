"""Process-local liveness and filesystem readiness checks."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

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


HealthProbe = Callable[[], bool | str]


@dataclass(slots=True)
class DependencyHealthService:
    """Liveness/readiness report with explicit configured dependency checks."""

    probes: Mapping[str, HealthProbe]
    required: tuple[str, ...]
    event_sink: EventSink = field(default_factory=NullEventSink)

    _DEPENDENCIES = (
        "database",
        "queue",
        "artifact_store",
        "lock_service",
        "secret_provider",
        "worker",
        "migrations",
    )

    @staticmethod
    def _run_probe(probe: HealthProbe) -> str:
        try:
            result = probe()
        except Exception as exc:
            return f"error:{type(exc).__name__}"
        if result is True:
            return "ok"
        if result is False:
            return "unavailable"
        return result if result else "unavailable"

    def check(self) -> HealthReport:
        checks: dict[str, str] = {}
        names = tuple(dict.fromkeys((*self._DEPENDENCIES, *self.probes.keys())))
        for name in names:
            probe = self.probes.get(name)
            checks[name] = "not_configured" if probe is None else self._run_probe(probe)
        ready = all(checks.get(name) == "ok" for name in self.required)
        report = HealthReport(live=True, ready=ready, checks=checks)
        self.event_sink.emit(
            StructuredEvent.create(
                "health.dependencies_checked",
                "health",
                severity="info" if ready else "error",
                attributes={"live": True, "ready": ready, **checks},
            )
        )
        return report
