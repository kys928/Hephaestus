"""Single-job worker boundary for local execution and adapter tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from hephaestus.infrastructure.observability import EventSink, NullEventSink, StructuredEvent

from .models import JobRecord, JobStatus
from .queue import JobQueue

JobHandler = Callable[[JobRecord], str | None]


@dataclass(slots=True)
class LocalWorker:
    worker_id: str
    queue: JobQueue
    handler: JobHandler
    lease_seconds: int = 60
    event_sink: EventSink = field(default_factory=NullEventSink)

    def run_once(self) -> JobRecord | None:
        self.event_sink.emit(
            StructuredEvent.create(
                "worker.polling", "local_worker", entity_id=self.worker_id
            )
        )
        leased = self.queue.lease_next(self.worker_id, self.lease_seconds)
        if leased is None:
            self.event_sink.emit(
                StructuredEvent.create("worker.idle", "local_worker", entity_id=self.worker_id)
            )
            return None
        running = self.queue.start(leased.job_id, self.worker_id)
        if running.status is JobStatus.CANCELLED:
            self.event_sink.emit(
                StructuredEvent.create(
                    "worker.job_cancelled",
                    "local_worker",
                    entity_id=self.worker_id,
                    attributes={"job_id": running.job_id},
                )
            )
            return running
        self.event_sink.emit(
            StructuredEvent.create(
                "worker.job_started",
                "local_worker",
                entity_id=self.worker_id,
                attributes={"job_id": running.job_id, "attempt_count": running.attempt_count},
            )
        )
        try:
            result_ref = self.handler(running)
        except Exception as exc:  # adapter boundary converts exceptions into non-secret references
            self.event_sink.emit(
                StructuredEvent.create(
                    "worker.job_failed",
                    "local_worker",
                    entity_id=self.worker_id,
                    severity="error",
                    attributes={"job_id": running.job_id, "error_type": type(exc).__name__},
                )
            )
            return self.queue.fail(
                running.job_id,
                self.worker_id,
                error_ref=f"exception:{type(exc).__name__}",
            )
        latest = self.queue.get(running.job_id)
        if latest is not None and latest.cancellation_requested:
            result = self.queue.acknowledge_cancellation(running.job_id, self.worker_id)
            event_type = "worker.job_cancelled"
        else:
            result = self.queue.complete(running.job_id, self.worker_id, result_ref)
            event_type = "worker.job_finished"
        self.event_sink.emit(
            StructuredEvent.create(
                event_type,
                "local_worker",
                entity_id=self.worker_id,
                attributes={"job_id": result.job_id, "status": result.status.value},
            )
        )
        return result
