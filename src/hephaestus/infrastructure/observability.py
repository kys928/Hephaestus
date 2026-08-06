"""Small structured-observability boundaries with local implementations."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


_SENSITIVE_FRAGMENTS = ("secret", "password", "token", "credential", "private_key")
_MAX_ATTRIBUTE_STRING = 512


def _safe_attributes(attributes: dict[str, object]) -> dict[str, object]:
    """Bound event attributes and redact values whose names imply secret material."""

    safe: dict[str, object] = {}
    for key, value in attributes.items():
        lowered = key.lower()
        if any(fragment in lowered for fragment in _SENSITIVE_FRAGMENTS):
            safe[key] = "[redacted]"
        elif isinstance(value, str):
            safe[key] = value[:_MAX_ATTRIBUTE_STRING]
        elif isinstance(value, (bool, int, float)) or value is None:
            safe[key] = value
        elif isinstance(value, (list, tuple)):
            safe[key] = [str(item)[:128] for item in value[:32]]
        else:
            safe[key] = str(value)[:_MAX_ATTRIBUTE_STRING]
    return safe


@dataclass(frozen=True, slots=True)
class StructuredEvent:
    event_id: str
    event_type: str
    component: str
    timestamp: datetime
    entity_id: str | None = None
    severity: str = "info"
    attributes: dict[str, object] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        event_type: str,
        component: str,
        *,
        entity_id: str | None = None,
        severity: str = "info",
        attributes: dict[str, object] | None = None,
        timestamp: datetime | None = None,
    ) -> "StructuredEvent":
        return cls(
            event_id=str(uuid4()),
            event_type=event_type,
            component=component,
            timestamp=timestamp or utc_now(),
            entity_id=entity_id,
            severity=severity,
            attributes=_safe_attributes(dict(attributes or {})),
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


class EventSink(Protocol):
    def emit(self, event: StructuredEvent) -> None: ...


def emit_safely(sink: EventSink, event: StructuredEvent) -> bool:
    """Observability must never roll back or counterfeit an infrastructure result."""

    try:
        sink.emit(event)
    except Exception:
        return False
    return True


class NullEventSink:
    def emit(self, event: StructuredEvent) -> None:
        del event


@dataclass(slots=True)
class InMemoryEventSink:
    events: list[StructuredEvent] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def emit(self, event: StructuredEvent) -> None:
        with self._lock:
            self.events.append(event)


@dataclass(slots=True)
class JsonLineEventSink:
    path: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def emit(self, event: StructuredEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":"))
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


@dataclass(slots=True)
class MetricsCollector:
    """In-process telemetry derived from events; not a durable metrics backend."""

    counters: dict[str, int] = field(default_factory=dict)
    observations: dict[str, list[float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def emit(self, event: StructuredEvent) -> None:
        with self._lock:
            self.counters[event.event_type] = self.counters.get(event.event_type, 0) + 1
            for key in ("queue_delay_seconds", "execution_duration_seconds"):
                value = event.attributes.get(key)
                if isinstance(value, (int, float)):
                    self.observations.setdefault(key, []).append(float(value))

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "counters": dict(self.counters),
                "observations": {key: list(values) for key, values in self.observations.items()},
            }


@dataclass(slots=True)
class CompositeEventSink:
    sinks: tuple[EventSink, ...]

    def emit(self, event: StructuredEvent) -> None:
        for sink in self.sinks:
            emit_safely(sink, event)


@dataclass(slots=True)
class OpenTelemetryEventSink:
    """Optional OpenTelemetry bridge using injected tracer and meter objects.

    The class deliberately imports no OpenTelemetry package. ``from_installed`` is
    the only dependency-loading path, so core imports remain lightweight.
    """

    tracer: Any
    meter: Any | None = None
    _counter: Any | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_installed(cls, instrumentation_name: str = "hephaestus.infrastructure") -> "OpenTelemetryEventSink":
        try:
            from opentelemetry import metrics, trace
        except ImportError as exc:  # pragma: no cover - optional dependency
            from .capabilities import OptionalCapabilityError

            raise OptionalCapabilityError(
                "OpenTelemetry support requires the 'telemetry' optional dependencies"
            ) from exc
        return cls(trace.get_tracer(instrumentation_name), metrics.get_meter(instrumentation_name))

    def emit(self, event: StructuredEvent) -> None:
        attributes = {
            "event.id": event.event_id,
            "event.type": event.event_type,
            "component": event.component,
            "severity": event.severity,
            **{f"hephaestus.{key}": value for key, value in event.attributes.items()},
        }
        if event.entity_id:
            attributes["entity.id"] = event.entity_id
        with self.tracer.start_as_current_span(event.event_type, attributes=attributes) as span:
            span.add_event(event.event_type, attributes=attributes)
        if self.meter is not None:
            if self._counter is None:
                self._counter = self.meter.create_counter(
                    "hephaestus.infrastructure.events",
                    description="Structured infrastructure events",
                )
            self._counter.add(
                1, {"event.type": event.event_type, "component": event.component}
            )
