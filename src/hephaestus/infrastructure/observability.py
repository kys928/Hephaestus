"""Small structured-observability boundaries with local implementations."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


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
            attributes=dict(attributes or {}),
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


class EventSink(Protocol):
    def emit(self, event: StructuredEvent) -> None: ...


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
            sink.emit(event)
