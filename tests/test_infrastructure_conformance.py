from __future__ import annotations

from datetime import datetime, timezone

import pytest

from hephaestus.jobs import InMemoryJobQueue, JobStatus, SQLiteJobQueue
from hephaestus.storage import (
    InMemoryLockProvider,
    JsonLineStateRepository,
    SQLiteLockProvider,
    SQLiteStateRepository,
)


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture(params=("memory", "sqlite"))
def conforming_queue(request, tmp_path):
    if request.param == "memory":
        return InMemoryJobQueue()
    return SQLiteJobQueue(tmp_path / "queue.sqlite3")


@pytest.fixture(params=("jsonl", "sqlite"))
def conforming_state_repository(request, tmp_path):
    if request.param == "jsonl":
        return JsonLineStateRepository(tmp_path / "jsonl")
    return SQLiteStateRepository(tmp_path / "state.sqlite3")


@pytest.fixture(params=("memory", "sqlite"))
def conforming_lock_provider(request, tmp_path):
    if request.param == "memory":
        return InMemoryLockProvider()
    return SQLiteLockProvider(tmp_path / "locks.sqlite3")


def test_queue_submission_lookup_idempotency_and_queued_cancellation_conformance(
    conforming_queue,
) -> None:
    first = conforming_queue.submit(
        "payload",
        "owner",
        run_id="run",
        experiment_id="experiment",
        idempotency_key="operation",
        job_id="job",
        now=NOW,
    )
    duplicate = conforming_queue.submit(
        "payload",
        "owner",
        run_id="run",
        experiment_id="experiment",
        idempotency_key="operation",
        now=NOW,
    )
    cancelled = conforming_queue.request_cancellation("job", now=NOW)

    assert duplicate.job_id == first.job_id
    assert conforming_queue.get("job").status is JobStatus.CANCELLED
    assert cancelled.cancellation_requested is True


def test_state_append_order_and_latest_lookup_conformance(
    conforming_state_repository,
) -> None:
    conforming_state_repository.append("records", {"id": "one", "value": 1})
    conforming_state_repository.append("records", {"id": "two", "value": 2})

    assert conforming_state_repository.all("records") == [
        {"id": "one", "value": 1},
        {"id": "two", "value": 2},
    ]
    assert conforming_state_repository.get_latest("records", "id", "two") == {
        "id": "two",
        "value": 2,
    }


def test_lock_owner_contention_renewal_and_release_conformance(
    conforming_lock_provider,
) -> None:
    lease = conforming_lock_provider.acquire("resource:one", "worker-a", 10, now=NOW)
    assert lease is not None
    assert conforming_lock_provider.acquire(
        "resource:one", "worker-b", 10, now=NOW
    ) is None
    renewed = conforming_lock_provider.heartbeat(lease, 20, now=NOW)
    assert renewed is not None
    assert conforming_lock_provider.release(renewed) is True
