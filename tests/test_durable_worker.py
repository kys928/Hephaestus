from __future__ import annotations

import threading
import time

from hephaestus.infrastructure.observability import InMemoryEventSink
from hephaestus.jobs import (
    DurableWorker,
    JobStatus,
    NonRetryableJobError,
    RetryableJobError,
    SQLiteJobQueue,
)


def test_worker_heartbeats_and_completes_with_fenced_lease(tmp_path) -> None:
    events = InMemoryEventSink()
    queue = SQLiteJobQueue(tmp_path / "jobs.sqlite3")
    queue.submit("payload", "owner", job_id="job")

    def handler(job, context):
        del context
        time.sleep(0.12)
        return f"result:{job.job_id}"

    worker = DurableWorker(
        "worker-1",
        queue,
        handler,
        lease_seconds=1,
        heartbeat_interval_seconds=0.05,
        event_sink=events,
    )
    result = worker.run_once()

    assert result.status is JobStatus.SUCCEEDED
    assert result.result_ref == "result:job"
    assert any(row["event_type"] == "worker.heartbeat" for row in queue.audit_records("job"))
    assert events.events[-1].event_type == "worker.job_finished"


def test_retryable_failure_requeues_then_succeeds(tmp_path) -> None:
    queue = SQLiteJobQueue(tmp_path / "jobs.sqlite3", maximum_attempts=2)
    queue.submit("payload", "owner", job_id="job")
    calls = 0

    def handler(_job, _context):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RetryableJobError("transient transport detail must not persist")
        return "result"

    worker = DurableWorker(
        "worker-1",
        queue,
        handler,
        lease_seconds=2,
        heartbeat_interval_seconds=0.1,
    )
    first = worker.run_once()
    second = worker.run_once()

    assert first.status is JobStatus.QUEUED
    assert second.status is JobStatus.SUCCEEDED
    assert "transient transport detail" not in str(queue.audit_records("job"))


def test_unknown_and_non_retryable_failures_dead_letter_without_secret_messages(tmp_path) -> None:
    queue = SQLiteJobQueue(tmp_path / "jobs.sqlite3")
    queue.submit("payload-a", "owner", job_id="unknown")
    queue.submit("payload-b", "owner", job_id="classified")

    def unknown(_job, _context):
        raise RuntimeError("private failure detail")

    result = DurableWorker(
        "worker-1",
        queue,
        unknown,
        lease_seconds=2,
        heartbeat_interval_seconds=0.1,
    ).run_once()
    assert result.status is JobStatus.DEAD_LETTER
    assert result.error_ref == "exception:RuntimeError"
    assert "private failure detail" not in str(result.to_dict())

    def classified(_job, _context):
        raise NonRetryableJobError("invalid payload body")

    result = DurableWorker(
        "worker-1",
        queue,
        classified,
        lease_seconds=2,
        heartbeat_interval_seconds=0.1,
    ).run_once()
    assert result.status is JobStatus.DEAD_LETTER
    assert result.error_ref == "exception:NonRetryableJobError"


def test_worker_cooperatively_acknowledges_persisted_cancellation(tmp_path) -> None:
    queue = SQLiteJobQueue(tmp_path / "jobs.sqlite3")
    queue.submit("payload", "owner", job_id="job")
    handler_started = threading.Event()

    def handler(_job, context):
        handler_started.set()
        while not context.cancellation_requested():
            time.sleep(0.01)
        context.raise_if_cancelled()

    worker = DurableWorker(
        "worker-1",
        queue,
        handler,
        lease_seconds=2,
        heartbeat_interval_seconds=0.1,
    )
    result_holder = []
    thread = threading.Thread(target=lambda: result_holder.append(worker.run_once()))
    thread.start()
    assert handler_started.wait(2)
    queue.request_cancellation("job")
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert result_holder[0].status is JobStatus.CANCELLED
    assert result_holder[0].cancellation_acknowledged is True


def test_shutdown_stops_new_leasing_and_emits_lifecycle(tmp_path) -> None:
    events = InMemoryEventSink()
    queue = SQLiteJobQueue(tmp_path / "jobs.sqlite3")
    worker = DurableWorker(
        "worker-1",
        queue,
        lambda _job, _context: "result",
        lease_seconds=2,
        heartbeat_interval_seconds=0.1,
        poll_interval_seconds=0.01,
        event_sink=events,
    )
    worker.run_forever(maximum_idle_polls=1)
    worker.request_shutdown()
    queue.submit("payload", "owner", job_id="job")

    assert worker.run_once() is None
    assert queue.get("job").status is JobStatus.QUEUED
    assert [event.event_type for event in events.events] == [
        "worker.started",
        "worker.idle",
        "worker.stopped",
    ]
