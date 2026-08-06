from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from hephaestus.storage import (
    LockLostError,
    LockTimeoutError,
    SQLiteLockProvider,
    SQLiteStateRepository,
    StateOperationConflict,
)


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_state_repository_orders_records_and_supports_latest_by_key(tmp_path) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.sqlite3")
    first = repository.append(
        "decisions",
        {"decision_id": "d-1", "value": 1},
        operation_id="operation-1",
        record_key="decision-1",
        timestamp=NOW,
    )
    duplicate = repository.append(
        "decisions",
        {"decision_id": "d-1", "value": 1},
        operation_id="operation-1",
        record_key="decision-1",
        timestamp=NOW + timedelta(seconds=1),
    )
    repository.append(
        "decisions",
        {"decision_id": "d-1", "value": 2},
        operation_id="operation-2",
        record_key="decision-1",
        timestamp=NOW + timedelta(seconds=2),
    )

    assert duplicate == first
    assert repository.all("decisions") == [
        {"decision_id": "d-1", "value": 1},
        {"decision_id": "d-1", "value": 2},
    ]
    assert repository.latest_by_key("decisions", "decision-1")["value"] == 2
    assert repository.get_latest("decisions", "decision_id", "d-1")["value"] == 2
    assert [row["sequence"] for row in repository.records("decisions")] == [1, 2]


def test_state_operation_id_conflict_and_transaction_rollback(tmp_path) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.sqlite3")
    repository.append("records", {"value": 1}, operation_id="operation-1")
    with pytest.raises(StateOperationConflict):
        repository.append("records", {"value": 2}, operation_id="operation-1")

    with pytest.raises(RuntimeError):
        with repository.transaction() as transaction:
            transaction.append("records", {"value": 3}, operation_id="operation-2")
            transaction.append("records", {"value": 4}, operation_id="operation-3")
            raise RuntimeError("roll back all writes")

    assert repository.all("records") == [{"value": 1}]


def test_state_repository_concurrent_appends_are_unique_and_durable(tmp_path) -> None:
    path = tmp_path / "state.sqlite3"
    repository = SQLiteStateRepository(path)

    def append(index: int) -> None:
        SQLiteStateRepository(path, initialize=False).append(
            "events",
            {"index": index},
            operation_id=f"operation-{index}",
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(append, range(100)))

    restarted = SQLiteStateRepository(path, initialize=False)
    assert len(restarted.all("events")) == 100
    assert {row["index"] for row in restarted.all("events")} == set(range(100))


def test_sqlite_lock_contends_renews_expires_and_detects_loss(tmp_path) -> None:
    path = tmp_path / "locks.sqlite3"
    first_provider = SQLiteLockProvider(path)
    second_provider = SQLiteLockProvider(path, initialize=False)
    first = first_provider.acquire("training:run-1", "worker-a", 5, now=NOW)

    assert first is not None and first.lease_token
    assert second_provider.acquire("training:run-1", "worker-b", 5, now=NOW) is None
    renewed = first_provider.heartbeat(first, 10, now=NOW + timedelta(seconds=1))
    assert renewed is not None and renewed.expires_at == NOW + timedelta(seconds=11)
    assert second_provider.acquire(
        "training:run-1", "worker-b", 5, now=NOW + timedelta(seconds=11)
    ) is not None
    with pytest.raises(LockLostError):
        first_provider.assert_owned(renewed, now=NOW + timedelta(seconds=11))
    assert first_provider.release(renewed) is False


def test_lock_context_is_exception_safe_and_timeout_is_explicit(tmp_path) -> None:
    provider = SQLiteLockProvider(tmp_path / "locks.sqlite3")
    with pytest.raises(RuntimeError):
        with provider.held("artifact:index", "worker-a", 10):
            raise RuntimeError("body failed")

    with provider.held("artifact:index", "worker-b", 10) as lease:
        assert lease.owner == "worker-b"
        with pytest.raises(LockTimeoutError):
            with provider.held(
                "artifact:index", "worker-c", 10, timeout_seconds=0.01
            ):
                pass


def test_state_and_lock_collection_validation(tmp_path) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.sqlite3")
    locks = SQLiteLockProvider(tmp_path / "state.sqlite3")
    with pytest.raises(ValueError):
        repository.append("../escape", {"value": 1})
    with pytest.raises(ValueError):
        locks.acquire("../escape", "worker", 10)
