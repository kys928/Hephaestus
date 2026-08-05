from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hephaestus.infrastructure.observability import InMemoryEventSink, MetricsCollector
from hephaestus.jobs import InMemoryJobQueue, JobStatus, LocalWorker
from hephaestus.jobs.queue import IdempotencyConflict, LeaseOwnershipError


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_submit_is_idempotent_and_rejects_key_reuse_for_another_payload() -> None:
    queue = InMemoryJobQueue()
    first = queue.submit(
        "artifact:payload-1",
        "planner",
        run_id="run-1",
        experiment_id="exp-1",
        idempotency_key="submit-1",
        job_id="job-1",
        now=NOW,
    )
    duplicate = queue.submit(
        "artifact:payload-1",
        "planner",
        run_id="run-1",
        experiment_id="exp-1",
        idempotency_key="submit-1",
        now=NOW + timedelta(seconds=1),
    )

    assert duplicate == first
    assert duplicate.job_id == "job-1"
    with pytest.raises(IdempotencyConflict):
        queue.submit(
            "artifact:different",
            "planner",
            idempotency_key="submit-1",
            now=NOW,
        )


def test_lease_owner_heartbeat_complete_and_duplicate_worker_protection() -> None:
    queue = InMemoryJobQueue()
    queue.submit("payload", "owner", job_id="job", now=NOW)
    leased = queue.lease_next("worker-a", 10, now=NOW + timedelta(seconds=1))

    assert leased is not None
    assert leased.status is JobStatus.LEASED
    assert leased.attempt_count == 1
    with pytest.raises(LeaseOwnershipError):
        queue.start("job", "worker-b", now=NOW + timedelta(seconds=2))

    running = queue.start("job", "worker-a", now=NOW + timedelta(seconds=2))
    renewed = queue.heartbeat("job", "worker-a", 20, now=NOW + timedelta(seconds=3))
    completed = queue.complete(
        "job", "worker-a", "artifact:result", now=NOW + timedelta(seconds=5)
    )

    assert running.status is JobStatus.RUNNING
    assert renewed.lease_expires_at == NOW + timedelta(seconds=23)
    assert completed.status is JobStatus.SUCCEEDED
    assert completed.result_ref == "artifact:result"
    assert completed.lease_owner is None


def test_lease_expiration_is_explicit_and_retry_preserves_identity() -> None:
    queue = InMemoryJobQueue()
    queue.submit("payload", "owner", job_id="job", now=NOW)
    queue.lease_next("worker-a", 5, now=NOW)

    expired = queue.expire_leases(now=NOW + timedelta(seconds=5))
    retried = queue.retry("job", now=NOW + timedelta(seconds=6))
    leased_again = queue.lease_next("worker-b", 5, now=NOW + timedelta(seconds=7))

    assert expired[0].status is JobStatus.EXPIRED
    assert expired[0].error_ref == "lease_expired"
    assert retried.job_id == "job"
    assert retried.status is JobStatus.QUEUED
    assert leased_again is not None and leased_again.attempt_count == 2


def test_cancellation_is_immediate_when_queued_and_signalled_when_running() -> None:
    queue = InMemoryJobQueue()
    queue.submit("queued", "owner", job_id="queued", now=NOW)
    cancelled = queue.request_cancellation("queued", now=NOW + timedelta(seconds=1))
    assert cancelled.status is JobStatus.CANCELLED

    queue.submit("running", "owner", job_id="running", now=NOW)
    queue.lease_next("worker", 20, now=NOW + timedelta(seconds=1))
    queue.start("running", "worker", now=NOW + timedelta(seconds=2))
    signalled = queue.request_cancellation("running", now=NOW + timedelta(seconds=3))
    acknowledged = queue.acknowledge_cancellation(
        "running", "worker", now=NOW + timedelta(seconds=4)
    )

    assert signalled.status is JobStatus.RUNNING
    assert signalled.cancellation_requested is True
    assert acknowledged.status is JobStatus.CANCELLED


def test_worker_emits_lifecycle_events_and_converts_handler_failure_to_reference() -> None:
    events = InMemoryEventSink()
    metrics = MetricsCollector()
    queue = InMemoryJobQueue(event_sink=metrics)
    queue.submit("payload", "owner")

    def fail(_job):
        raise RuntimeError("sensitive detail must not be persisted")

    result = LocalWorker("worker", queue, fail, event_sink=events).run_once()

    assert result is not None
    assert result.status is JobStatus.FAILED
    assert result.error_ref == "exception:RuntimeError"
    assert "sensitive detail" not in str(result.to_dict())
    assert [event.event_type for event in events.events] == [
        "worker.polling",
        "worker.job_started",
        "worker.job_failed",
    ]
    assert metrics.snapshot()["counters"]["job.failed"] == 1
