"""Multi-process worker loop for fenced durable queues."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable

from hephaestus.infrastructure.observability import (
    EventSink,
    NullEventSink,
    StructuredEvent,
    emit_safely,
)

from .models import JobRecord, JobStatus
from .queue import LeaseOwnershipError
from .sqlite import SQLiteJobQueue


class RetryableJobError(RuntimeError):
    """A transport/execution failure that may be retried within the attempt budget."""


class NonRetryableJobError(RuntimeError):
    """A caller-classified failure that must move directly to dead letter."""


class CooperativeCancellation(RuntimeError):
    """Raised by a handler after observing a persisted cancellation request."""


class WorkerShutdown(RuntimeError):
    """Raised by a handler that elects to hand work back after shutdown is requested."""


@dataclass(slots=True)
class JobExecutionContext:
    queue: SQLiteJobQueue
    job_id: str
    shutdown_event: threading.Event

    def cancellation_requested(self) -> bool:
        record = self.queue.get(self.job_id)
        return bool(record and record.cancellation_requested)

    def shutdown_requested(self) -> bool:
        return self.shutdown_event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancellation_requested():
            raise CooperativeCancellation(self.job_id)


DurableJobHandler = Callable[[JobRecord, JobExecutionContext], str | None]


@dataclass(slots=True)
class DurableWorker:
    """Lease-fenced worker with heartbeat, bounded polling, and graceful shutdown.

    The handler receives only a persisted payload reference. It must resolve and
    validate that reference through the configured artifact boundary. Shutdown does
    not invent a domain cancellation decision: a cooperative handler may return,
    observe an already persisted cancellation request, or raise ``WorkerShutdown``
    and let the lease expire for clean recovery.
    """

    worker_id: str
    queue: SQLiteJobQueue
    handler: DurableJobHandler
    lease_seconds: int = 60
    poll_interval_seconds: float = 1.0
    heartbeat_interval_seconds: float | None = None
    event_sink: EventSink = field(default_factory=NullEventSink)
    shutdown_event: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        if not self.worker_id.strip():
            raise ValueError("worker_id must not be empty")
        if self.lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        interval = self.heartbeat_interval_seconds
        if interval is None:
            interval = max(0.05, self.lease_seconds / 3)
            self.heartbeat_interval_seconds = interval
        if interval <= 0 or interval >= self.lease_seconds:
            raise ValueError("heartbeat interval must be positive and shorter than the lease")

    def _emit(
        self,
        event_type: str,
        *,
        severity: str = "info",
        attributes: dict[str, object] | None = None,
    ) -> None:
        emit_safely(
            self.event_sink,
            StructuredEvent.create(
                event_type,
                "durable_worker",
                entity_id=self.worker_id,
                severity=severity,
                attributes=attributes,
            ),
        )

    def request_shutdown(self) -> None:
        self.shutdown_event.set()

    def _heartbeat_loop(
        self,
        job: JobRecord,
        stop: threading.Event,
        ownership_lost: threading.Event,
    ) -> None:
        interval = float(self.heartbeat_interval_seconds or 0)
        while not stop.wait(interval):
            try:
                self.queue.heartbeat(
                    job.job_id,
                    self.worker_id,
                    self.lease_seconds,
                    lease_token=job.lease_token,
                )
            except Exception as exc:  # heartbeat failure must fence terminal persistence
                ownership_lost.set()
                self._emit(
                    "worker.heartbeat_failed",
                    severity="error",
                    attributes={
                        "job_id": job.job_id,
                        "error_type": type(exc).__name__,
                    },
                )
                return

    def run_once(self) -> JobRecord | None:
        if self.shutdown_event.is_set():
            return None
        leased = self.queue.lease_next(self.worker_id, self.lease_seconds)
        if leased is None:
            self._emit("worker.idle")
            return None
        if not leased.lease_token:
            raise LeaseOwnershipError("durable queue returned an unfenced lease")
        running = self.queue.start(
            leased.job_id,
            self.worker_id,
            lease_token=leased.lease_token,
        )
        if running.status is JobStatus.CANCELLED:
            self._emit(
                "worker.job_cancelled",
                attributes={"job_id": running.job_id, "status": running.status.value},
            )
            return running

        self._emit(
            "worker.job_started",
            attributes={"job_id": running.job_id, "attempt_count": running.attempt_count},
        )
        heartbeat_stop = threading.Event()
        ownership_lost = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat_loop,
            args=(running, heartbeat_stop, ownership_lost),
            name=f"hephaestus-heartbeat-{self.worker_id}",
            daemon=True,
        )
        heartbeat.start()
        context = JobExecutionContext(self.queue, running.job_id, self.shutdown_event)
        outcome: tuple[str, str | None, bool] = ("success", None, False)
        try:
            result_ref = self.handler(running, context)
            outcome = ("success", result_ref, False)
        except CooperativeCancellation:
            outcome = ("cancel", None, False)
        except WorkerShutdown:
            outcome = ("handoff", None, True)
        except RetryableJobError as exc:
            outcome = ("failure", f"exception:{type(exc).__name__}", True)
        except NonRetryableJobError as exc:
            outcome = ("failure", f"exception:{type(exc).__name__}", False)
        except Exception as exc:  # unknown failures are normalized and not retried implicitly
            outcome = ("failure", f"exception:{type(exc).__name__}", False)
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=max(1.0, float(self.heartbeat_interval_seconds or 0) * 2))

        if ownership_lost.is_set() or outcome[0] == "handoff":
            self._emit(
                "worker.ownership_lost" if ownership_lost.is_set() else "worker.handoff",
                severity="warning",
                attributes={"job_id": running.job_id},
            )
            return self.queue.get(running.job_id)

        latest = self.queue.get(running.job_id)
        if latest is not None and latest.cancellation_requested:
            result = self.queue.acknowledge_cancellation(
                running.job_id,
                self.worker_id,
                lease_token=running.lease_token,
            )
            event_type = "worker.job_cancelled"
        elif outcome[0] == "cancel":
            # A cancellation exception without persisted intent cannot manufacture
            # authorization; leave the lease for explicit recovery.
            self._emit(
                "worker.cancellation_unconfirmed",
                severity="warning",
                attributes={"job_id": running.job_id},
            )
            return latest
        elif outcome[0] == "failure":
            result = self.queue.fail(
                running.job_id,
                self.worker_id,
                outcome[1] or "exception:Unknown",
                retryable=outcome[2],
                lease_token=running.lease_token,
            )
            if result.status is JobStatus.FAILED and outcome[2]:
                result = self.queue.retry(result.job_id)
                event_type = "worker.job_retry_queued"
            else:
                event_type = (
                    "worker.job_dead_lettered"
                    if result.status is JobStatus.DEAD_LETTER
                    else "worker.job_failed"
                )
        else:
            result = self.queue.complete(
                running.job_id,
                self.worker_id,
                outcome[1],
                lease_token=running.lease_token,
            )
            event_type = "worker.job_finished"
        self._emit(
            event_type,
            severity="error" if result.status is JobStatus.DEAD_LETTER else "info",
            attributes={"job_id": result.job_id, "status": result.status.value},
        )
        return result

    def run_forever(self, *, maximum_idle_polls: int | None = None) -> None:
        if maximum_idle_polls is not None and maximum_idle_polls < 0:
            raise ValueError("maximum_idle_polls must not be negative")
        self._emit("worker.started")
        idle_polls = 0
        try:
            while not self.shutdown_event.is_set():
                result = self.run_once()
                if result is None:
                    idle_polls += 1
                    if maximum_idle_polls is not None and idle_polls >= maximum_idle_polls:
                        break
                    self.shutdown_event.wait(self.poll_interval_seconds)
                else:
                    idle_polls = 0
        finally:
            self._emit("worker.stopped")
