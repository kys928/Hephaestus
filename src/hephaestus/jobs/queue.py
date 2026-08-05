"""Job queue protocol and honest process-local implementation."""

from __future__ import annotations

import threading
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Protocol
from uuid import uuid4

from hephaestus.infrastructure.observability import EventSink, NullEventSink, StructuredEvent

from .models import JobRecord, JobStatus, TERMINAL_JOB_STATUSES


class JobQueueError(RuntimeError):
    pass


class JobNotFound(JobQueueError):
    pass


class InvalidJobTransition(JobQueueError):
    pass


class LeaseOwnershipError(JobQueueError):
    pass


class IdempotencyConflict(JobQueueError):
    pass


class JobQueue(Protocol):
    def submit(
        self,
        payload_ref: str,
        owner_id: str,
        *,
        run_id: str | None = None,
        experiment_id: str | None = None,
        idempotency_key: str | None = None,
        job_id: str | None = None,
        now: datetime | None = None,
    ) -> JobRecord: ...

    def get(self, job_id: str) -> JobRecord | None: ...

    def lease_next(
        self, worker_id: str, lease_seconds: int, *, now: datetime | None = None
    ) -> JobRecord | None: ...

    def heartbeat(
        self, job_id: str, worker_id: str, lease_seconds: int, *, now: datetime | None = None
    ) -> JobRecord: ...

    def start(
        self, job_id: str, worker_id: str, *, now: datetime | None = None
    ) -> JobRecord: ...

    def complete(
        self,
        job_id: str,
        worker_id: str,
        result_ref: str | None,
        *,
        now: datetime | None = None,
    ) -> JobRecord: ...

    def fail(
        self,
        job_id: str,
        worker_id: str,
        error_ref: str,
        *,
        now: datetime | None = None,
    ) -> JobRecord: ...

    def request_cancellation(self, job_id: str, *, now: datetime | None = None) -> JobRecord: ...

    def acknowledge_cancellation(
        self, job_id: str, worker_id: str, *, now: datetime | None = None
    ) -> JobRecord: ...


def _timestamp(now: datetime | None) -> datetime:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("job timestamps must be timezone-aware")
    return value


class InMemoryJobQueue:
    """Thread-safe in one process; it provides no cross-process durability or ordering."""

    def __init__(self, event_sink: EventSink | None = None) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._idempotency: dict[str, str] = {}
        self._lock = threading.RLock()
        self._events = event_sink or NullEventSink()

    def _emit(
        self,
        event_type: str,
        record: JobRecord,
        *,
        severity: str = "info",
        attributes: dict[str, object] | None = None,
    ) -> None:
        details: dict[str, object] = {
            "status": record.status.value,
            "attempt_count": record.attempt_count,
            "owner_id": record.owner_id,
        }
        details.update(attributes or {})
        self._events.emit(
            StructuredEvent.create(
                event_type,
                "job_queue",
                entity_id=record.job_id,
                severity=severity,
                attributes=details,
                timestamp=record.updated_at,
            )
        )

    def _required(self, job_id: str) -> JobRecord:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise JobNotFound(job_id) from exc

    @staticmethod
    def _validate_lease_seconds(lease_seconds: int) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")

    @staticmethod
    def _assert_owner(record: JobRecord, worker_id: str) -> None:
        if record.lease_owner != worker_id:
            raise LeaseOwnershipError(
                f"job {record.job_id} is leased by {record.lease_owner!r}, not {worker_id!r}"
            )

    def submit(
        self,
        payload_ref: str,
        owner_id: str,
        *,
        run_id: str | None = None,
        experiment_id: str | None = None,
        idempotency_key: str | None = None,
        job_id: str | None = None,
        now: datetime | None = None,
    ) -> JobRecord:
        timestamp = _timestamp(now)
        with self._lock:
            if idempotency_key and idempotency_key in self._idempotency:
                existing = self._required(self._idempotency[idempotency_key])
                identity = (payload_ref, owner_id, run_id, experiment_id)
                existing_identity = (
                    existing.payload_ref,
                    existing.owner_id,
                    existing.run_id,
                    existing.experiment_id,
                )
                if identity != existing_identity:
                    raise IdempotencyConflict(
                        f"idempotency key {idempotency_key!r} was already used for another job"
                    )
                return existing
            resolved_job_id = job_id or str(uuid4())
            if resolved_job_id in self._jobs:
                raise IdempotencyConflict(f"job ID already exists: {resolved_job_id}")
            record = JobRecord(
                job_id=resolved_job_id,
                payload_ref=payload_ref,
                owner_id=owner_id,
                run_id=run_id,
                experiment_id=experiment_id,
                idempotency_key=idempotency_key,
                status=JobStatus.QUEUED,
                attempt_count=0,
                created_at=timestamp,
                updated_at=timestamp,
            )
            self._jobs[record.job_id] = record
            if idempotency_key:
                self._idempotency[idempotency_key] = record.job_id
            self._emit("job.queued", record)
            return record

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[JobRecord]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda record: (record.created_at, record.job_id))

    def expire_leases(self, *, now: datetime | None = None) -> list[JobRecord]:
        timestamp = _timestamp(now)
        expired: list[JobRecord] = []
        with self._lock:
            for job_id, record in list(self._jobs.items()):
                if (
                    record.status in {JobStatus.LEASED, JobStatus.RUNNING}
                    and record.lease_expires_at is not None
                    and record.lease_expires_at <= timestamp
                ):
                    updated = replace(
                        record,
                        status=JobStatus.EXPIRED,
                        updated_at=timestamp,
                        finished_at=timestamp,
                        lease_owner=None,
                        lease_expires_at=None,
                        error_ref="lease_expired",
                    )
                    self._jobs[job_id] = updated
                    expired.append(updated)
                    self._emit("job.expired", updated, severity="warning")
        return expired

    def lease_next(
        self, worker_id: str, lease_seconds: int, *, now: datetime | None = None
    ) -> JobRecord | None:
        self._validate_lease_seconds(lease_seconds)
        timestamp = _timestamp(now)
        with self._lock:
            self.expire_leases(now=timestamp)
            queued = [record for record in self._jobs.values() if record.status is JobStatus.QUEUED]
            if not queued:
                return None
            record = min(queued, key=lambda item: (item.created_at, item.job_id))
            updated = replace(
                record,
                status=JobStatus.LEASED,
                attempt_count=record.attempt_count + 1,
                lease_owner=worker_id,
                lease_expires_at=timestamp + timedelta(seconds=lease_seconds),
                updated_at=timestamp,
            )
            self._jobs[record.job_id] = updated
            queue_delay = max(0.0, (timestamp - record.created_at).total_seconds())
            self._emit(
                "job.leased",
                updated,
                attributes={"worker_id": worker_id, "queue_delay_seconds": queue_delay},
            )
            return updated

    def start(self, job_id: str, worker_id: str, *, now: datetime | None = None) -> JobRecord:
        timestamp = _timestamp(now)
        with self._lock:
            self.expire_leases(now=timestamp)
            record = self._required(job_id)
            if record.status is JobStatus.EXPIRED:
                raise InvalidJobTransition(f"cannot start expired job {job_id}")
            self._assert_owner(record, worker_id)
            if record.status is not JobStatus.LEASED:
                raise InvalidJobTransition(f"cannot start {record.status.value} job {job_id}")
            if record.cancellation_requested:
                return self.acknowledge_cancellation(job_id, worker_id, now=timestamp)
            updated = replace(
                record,
                status=JobStatus.RUNNING,
                started_at=record.started_at or timestamp,
                updated_at=timestamp,
            )
            self._jobs[job_id] = updated
            self._emit("job.running", updated, attributes={"worker_id": worker_id})
            return updated

    def heartbeat(
        self, job_id: str, worker_id: str, lease_seconds: int, *, now: datetime | None = None
    ) -> JobRecord:
        self._validate_lease_seconds(lease_seconds)
        timestamp = _timestamp(now)
        with self._lock:
            self.expire_leases(now=timestamp)
            record = self._required(job_id)
            self._assert_owner(record, worker_id)
            if record.status not in {JobStatus.LEASED, JobStatus.RUNNING}:
                raise InvalidJobTransition(f"cannot heartbeat {record.status.value} job {job_id}")
            updated = replace(
                record,
                lease_expires_at=timestamp + timedelta(seconds=lease_seconds),
                updated_at=timestamp,
            )
            self._jobs[job_id] = updated
            self._emit("worker.heartbeat", updated, attributes={"worker_id": worker_id})
            return updated

    def request_cancellation(self, job_id: str, *, now: datetime | None = None) -> JobRecord:
        timestamp = _timestamp(now)
        with self._lock:
            record = self._required(job_id)
            if record.status in TERMINAL_JOB_STATUSES:
                return record
            if record.status is JobStatus.QUEUED:
                updated = replace(
                    record,
                    status=JobStatus.CANCELLED,
                    cancellation_requested=True,
                    updated_at=timestamp,
                    finished_at=timestamp,
                )
                self._jobs[job_id] = updated
                self._emit("job.cancelled", updated)
                return updated
            updated = replace(record, cancellation_requested=True, updated_at=timestamp)
            self._jobs[job_id] = updated
            self._emit("job.cancellation_requested", updated, severity="warning")
            return updated

    def acknowledge_cancellation(
        self, job_id: str, worker_id: str, *, now: datetime | None = None
    ) -> JobRecord:
        timestamp = _timestamp(now)
        with self._lock:
            record = self._required(job_id)
            self._assert_owner(record, worker_id)
            if not record.cancellation_requested:
                raise InvalidJobTransition(f"job {job_id} has no cancellation request")
            if record.status not in {JobStatus.LEASED, JobStatus.RUNNING}:
                raise InvalidJobTransition(f"cannot cancel {record.status.value} job {job_id}")
            updated = replace(
                record,
                status=JobStatus.CANCELLED,
                updated_at=timestamp,
                finished_at=timestamp,
                lease_owner=None,
                lease_expires_at=None,
            )
            self._jobs[job_id] = updated
            self._emit(
                "job.cancelled",
                updated,
                attributes=self._execution_duration_attributes(updated, timestamp),
            )
            return updated

    @staticmethod
    def _execution_duration_attributes(record: JobRecord, now: datetime) -> dict[str, object]:
        if record.started_at is None:
            return {}
        return {"execution_duration_seconds": max(0.0, (now - record.started_at).total_seconds())}

    def complete(
        self, job_id: str, worker_id: str, result_ref: str | None, *, now: datetime | None = None
    ) -> JobRecord:
        timestamp = _timestamp(now)
        with self._lock:
            record = self._required(job_id)
            self._assert_owner(record, worker_id)
            if record.status is not JobStatus.RUNNING:
                raise InvalidJobTransition(f"cannot complete {record.status.value} job {job_id}")
            if record.cancellation_requested:
                return self.acknowledge_cancellation(job_id, worker_id, now=timestamp)
            updated = replace(
                record,
                status=JobStatus.SUCCEEDED,
                result_ref=result_ref,
                updated_at=timestamp,
                finished_at=timestamp,
                lease_owner=None,
                lease_expires_at=None,
            )
            self._jobs[job_id] = updated
            self._emit(
                "job.succeeded",
                updated,
                attributes=self._execution_duration_attributes(updated, timestamp),
            )
            return updated

    def fail(
        self, job_id: str, worker_id: str, error_ref: str, *, now: datetime | None = None
    ) -> JobRecord:
        timestamp = _timestamp(now)
        with self._lock:
            record = self._required(job_id)
            self._assert_owner(record, worker_id)
            if record.status is not JobStatus.RUNNING:
                raise InvalidJobTransition(f"cannot fail {record.status.value} job {job_id}")
            if record.cancellation_requested:
                return self.acknowledge_cancellation(job_id, worker_id, now=timestamp)
            updated = replace(
                record,
                status=JobStatus.FAILED,
                error_ref=error_ref,
                updated_at=timestamp,
                finished_at=timestamp,
                lease_owner=None,
                lease_expires_at=None,
            )
            self._jobs[job_id] = updated
            self._emit(
                "job.failed",
                updated,
                severity="error",
                attributes=self._execution_duration_attributes(updated, timestamp),
            )
            return updated

    def retry(self, job_id: str, *, now: datetime | None = None) -> JobRecord:
        timestamp = _timestamp(now)
        with self._lock:
            record = self._required(job_id)
            if record.status not in {JobStatus.FAILED, JobStatus.EXPIRED}:
                raise InvalidJobTransition(f"cannot retry {record.status.value} job {job_id}")
            updated = replace(
                record,
                status=JobStatus.QUEUED,
                updated_at=timestamp,
                lease_owner=None,
                lease_expires_at=None,
                started_at=None,
                finished_at=None,
                result_ref=None,
                error_ref=None,
                cancellation_requested=False,
            )
            self._jobs[job_id] = updated
            self._emit("job.retry_queued", updated, severity="warning")
            return updated
