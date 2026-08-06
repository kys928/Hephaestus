from __future__ import annotations

import multiprocessing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from hephaestus.jobs import JobStatus, SQLiteJobQueue
from hephaestus.jobs.queue import IdempotencyConflict, LeaseOwnershipError


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _competing_lease(path: str, worker_id: str, start, output) -> None:
    queue = SQLiteJobQueue(Path(path), initialize=False)
    start.wait()
    leased = queue.lease_next(worker_id, 30, now=NOW + timedelta(seconds=1))
    output.put(None if leased is None else (leased.job_id, leased.lease_owner))


def test_submit_is_durable_idempotent_and_conflict_safe(tmp_path) -> None:
    path = tmp_path / "jobs.sqlite3"
    first_queue = SQLiteJobQueue(path)
    first = first_queue.submit(
        "sha256:" + "1" * 64,
        "planner",
        run_id="run-1",
        experiment_id="exp-1",
        idempotency_key="operation-1",
        job_id="job-1",
        now=NOW,
    )

    restarted = SQLiteJobQueue(path, initialize=False)
    duplicate = restarted.submit(
        first.payload_ref,
        "planner",
        run_id="run-1",
        experiment_id="exp-1",
        idempotency_key="operation-1",
        now=NOW + timedelta(seconds=1),
    )

    assert duplicate.job_id == "job-1"
    assert restarted.get("job-1") == first
    with pytest.raises(IdempotencyConflict):
        restarted.submit("different", "planner", idempotency_key="operation-1", now=NOW)


def test_competing_processes_cannot_double_lease(tmp_path) -> None:
    path = tmp_path / "jobs.sqlite3"
    queue = SQLiteJobQueue(path)
    queue.submit("payload", "owner", job_id="job", now=NOW)
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    output = context.Queue()
    processes = [
        context.Process(target=_competing_lease, args=(str(path), f"worker-{index}", start, output))
        for index in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    results = [output.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert sum(result is not None for result in results) == 1
    assert queue.get("job").attempt_count == 1


def test_expired_lease_recovers_and_fencing_rejects_late_worker(tmp_path) -> None:
    queue = SQLiteJobQueue(tmp_path / "jobs.sqlite3")
    queue.submit("payload", "owner", job_id="job", now=NOW)
    first = queue.lease_next("worker-a", 5, now=NOW)
    assert first is not None and first.lease_token
    queue.start("job", "worker-a", lease_token=first.lease_token, now=NOW)

    recovered = queue.recover_expired_leases(now=NOW + timedelta(seconds=5))
    assert recovered[0].status is JobStatus.QUEUED
    second = queue.lease_next("worker-b", 5, now=NOW + timedelta(seconds=6))
    assert second is not None and second.lease_token != first.lease_token
    queue.start("job", "worker-b", lease_token=second.lease_token, now=NOW + timedelta(seconds=6))

    with pytest.raises(LeaseOwnershipError):
        queue.complete(
            "job",
            "worker-a",
            "late-result",
            lease_token=first.lease_token,
            now=NOW + timedelta(seconds=7),
        )
    completed = queue.complete(
        "job",
        "worker-b",
        "result",
        lease_token=second.lease_token,
        now=NOW + timedelta(seconds=7),
    )
    assert completed.status is JobStatus.SUCCEEDED


def test_heartbeat_and_persisted_cancellation_acknowledgement(tmp_path) -> None:
    queue = SQLiteJobQueue(tmp_path / "jobs.sqlite3")
    queue.submit("payload", "owner", job_id="job", now=NOW)
    lease = queue.lease_next("worker", 5, now=NOW)
    assert lease is not None
    queue.start("job", "worker", lease_token=lease.lease_token, now=NOW)
    renewed = queue.heartbeat(
        "job",
        "worker",
        10,
        lease_token=lease.lease_token,
        now=NOW + timedelta(seconds=2),
    )
    assert renewed.lease_expires_at == NOW + timedelta(seconds=12)
    signalled = queue.request_cancellation("job", now=NOW + timedelta(seconds=3))
    assert signalled.cancellation_requested is True
    cancelled = queue.acknowledge_cancellation(
        "job",
        "worker",
        lease_token=lease.lease_token,
        now=NOW + timedelta(seconds=4),
    )
    assert cancelled.status is JobStatus.CANCELLED
    assert cancelled.cancellation_acknowledged is True


def test_retry_attempt_budget_dead_letter_and_audited_replay(tmp_path) -> None:
    queue = SQLiteJobQueue(tmp_path / "jobs.sqlite3", maximum_attempts=2)
    queue.submit("payload", "owner", job_id="job", now=NOW)

    first = queue.lease_next("worker", 10, now=NOW)
    queue.start("job", "worker", lease_token=first.lease_token, now=NOW)
    failed = queue.fail(
        "job",
        "worker",
        "error:first",
        lease_token=first.lease_token,
        now=NOW + timedelta(seconds=1),
    )
    assert failed.status is JobStatus.FAILED
    queue.retry("job", now=NOW + timedelta(seconds=2))

    second = queue.lease_next("worker", 10, now=NOW + timedelta(seconds=3))
    queue.start(
        "job", "worker", lease_token=second.lease_token, now=NOW + timedelta(seconds=3)
    )
    dead = queue.fail(
        "job",
        "worker",
        "error:second",
        lease_token=second.lease_token,
        now=NOW + timedelta(seconds=4),
    )
    assert dead.status is JobStatus.DEAD_LETTER
    assert dead.dead_letter_reason == "maximum_attempts_exceeded"

    replayed = queue.replay_dead_letter(
        "job",
        actor="operator-1",
        reason="dependency was repaired",
        now=NOW + timedelta(seconds=5),
    )
    assert replayed.status is JobStatus.QUEUED
    assert replayed.replay_count == 1
    replay_audit = [
        row for row in queue.audit_records("job") if row["event_type"] == "job.dead_letter_replayed"
    ]
    assert replay_audit[0]["actor"] == "operator-1"
    assert replay_audit[0]["evidence"]["dead_letter_reason"] == "maximum_attempts_exceeded"


def test_non_retryable_and_malformed_payloads_dead_letter(tmp_path) -> None:
    queue = SQLiteJobQueue(tmp_path / "jobs.sqlite3")
    queue.submit("bad\nref", "owner", job_id="malformed", now=NOW)
    queue.submit("payload", "owner", job_id="non-retryable", now=NOW)

    leased = queue.lease_next("worker", 10, now=NOW)
    assert queue.get("malformed").status is JobStatus.DEAD_LETTER
    assert leased is not None and leased.job_id == "non-retryable"
    queue.start(leased.job_id, "worker", lease_token=leased.lease_token, now=NOW)
    dead = queue.fail(
        leased.job_id,
        "worker",
        "error:invalid",
        retryable=False,
        lease_token=leased.lease_token,
        now=NOW + timedelta(seconds=1),
    )
    assert dead.status is JobStatus.DEAD_LETTER
    assert dead.dead_letter_reason == "non_retryable_failure"


def test_terminal_and_dead_letter_state_survive_restart(tmp_path) -> None:
    path = tmp_path / "jobs.sqlite3"
    queue = SQLiteJobQueue(path)
    queue.submit("payload", "owner", job_id="job", now=NOW)
    lease = queue.lease_next("worker", 10, now=NOW)
    queue.start("job", "worker", lease_token=lease.lease_token, now=NOW)
    queue.complete("job", "worker", "result", lease_token=lease.lease_token, now=NOW)

    restarted = SQLiteJobQueue(path, initialize=False)
    assert restarted.get("job").status is JobStatus.SUCCEEDED
    assert restarted.lease_next("other", 10, now=NOW + timedelta(seconds=1)) is None
