"""Transactional SQLite state repository and cross-process lease locks."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from hephaestus.infrastructure.observability import (
    EventSink,
    NullEventSink,
    StructuredEvent,
    emit_safely,
)

from .base import LockLease


SQLITE_STATE_SCHEMA_VERSION = 1
SQLITE_LOCK_SCHEMA_VERSION = 1
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")


class StateOperationConflict(RuntimeError):
    pass


class LockTimeoutError(TimeoutError):
    pass


class LockLostError(RuntimeError):
    pass


def _timestamp(value: datetime | None = None) -> datetime:
    resolved = value or datetime.now(timezone.utc)
    if resolved.tzinfo is None or resolved.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return resolved.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return _timestamp(value).isoformat()


def _validate_name(value: str, kind: str) -> str:
    if not _NAME.fullmatch(value):
        raise ValueError(f"invalid {kind}: {value!r}")
    return value


def _canonical_record(record: dict[str, object]) -> str:
    if not isinstance(record, dict):
        raise TypeError("state record must be a JSON object")
    try:
        encoded = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("state record must be finite JSON-safe data") from exc
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ValueError("state record must be a JSON object")
    return encoded


class _SQLiteDatabase:
    def __init__(self, path: Path, busy_timeout_seconds: float) -> None:
        self.path = Path(path)
        self.busy_timeout_seconds = busy_timeout_seconds

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout={int(self.busy_timeout_seconds * 1000)}")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()


class SQLiteStateTransaction:
    def __init__(self, repository: "SQLiteStateRepository", connection: sqlite3.Connection) -> None:
        self.repository = repository
        self.connection = connection

    def append(
        self,
        collection: str,
        record: dict[str, object],
        *,
        operation_id: str | None = None,
        record_key: str | None = None,
        timestamp: datetime | None = None,
    ) -> dict[str, object]:
        return self.repository._append(
            self.connection,
            collection,
            record,
            operation_id=operation_id,
            record_key=record_key,
            timestamp=timestamp,
        )


class SQLiteStateRepository:
    """Ordered JSON-safe records with idempotent operation IDs and transactions."""

    def __init__(
        self,
        path: Path,
        *,
        busy_timeout_seconds: float = 5.0,
        event_sink: EventSink | None = None,
        initialize: bool = True,
    ) -> None:
        if busy_timeout_seconds < 0:
            raise ValueError("busy_timeout_seconds must not be negative")
        self._db = _SQLiteDatabase(path, busy_timeout_seconds)
        self.event_sink = event_sink or NullEventSink()
        if initialize:
            self.initialize()

    @property
    def path(self) -> Path:
        return self._db.path

    def initialize(self) -> None:
        with self._db.transaction() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS infrastructure_schema_migrations (
                    component TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    applied_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS state_records (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    collection TEXT NOT NULL,
                    operation_id TEXT NOT NULL UNIQUE,
                    record_key TEXT,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS state_collection_order
                    ON state_records(collection, sequence);
                CREATE INDEX IF NOT EXISTS state_latest_key
                    ON state_records(collection, record_key, sequence DESC);
                """
            )
            connection.execute(
                """
                INSERT INTO infrastructure_schema_migrations(component, version, applied_at)
                VALUES('state_repository', ?, ?)
                ON CONFLICT(component) DO UPDATE SET
                    version=excluded.version, applied_at=excluded.applied_at
                """,
                (SQLITE_STATE_SCHEMA_VERSION, _iso(_timestamp())),
            )

    def schema_version(self) -> int:
        with self._db.connect() as connection:
            row = connection.execute(
                "SELECT version FROM infrastructure_schema_migrations WHERE component='state_repository'"
            ).fetchone()
        return 0 if row is None else int(row["version"])

    def ready(self) -> bool:
        try:
            with self._db.connect() as connection:
                connection.execute("SELECT 1 FROM state_records LIMIT 1").fetchone()
            return self.schema_version() == SQLITE_STATE_SCHEMA_VERSION
        except sqlite3.Error:
            return False

    @staticmethod
    def _envelope(row: sqlite3.Row) -> dict[str, object]:
        return {
            "sequence": int(row["sequence"]),
            "collection": str(row["collection"]),
            "operation_id": str(row["operation_id"]),
            "record_key": row["record_key"],
            "created_at": str(row["created_at"]),
            "record": json.loads(row["payload_json"]),
        }

    def _append(
        self,
        connection: sqlite3.Connection,
        collection: str,
        record: dict[str, object],
        *,
        operation_id: str | None,
        record_key: str | None,
        timestamp: datetime | None,
    ) -> dict[str, object]:
        _validate_name(collection, "state collection")
        if record_key is not None:
            _validate_name(record_key, "record key")
        operation = operation_id or str(uuid4())
        _validate_name(operation, "operation ID")
        payload = _canonical_record(record)
        created_at = _timestamp(timestamp)
        existing = connection.execute(
            "SELECT * FROM state_records WHERE operation_id=?", (operation,)
        ).fetchone()
        if existing is not None:
            if (
                existing["collection"] != collection
                or existing["record_key"] != record_key
                or existing["payload_json"] != payload
            ):
                raise StateOperationConflict(
                    f"operation ID {operation!r} was already used for another record"
                )
            return self._envelope(existing)
        connection.execute(
            """
            INSERT INTO state_records(collection, operation_id, record_key, created_at, payload_json)
            VALUES(?, ?, ?, ?, ?)
            """,
            (collection, operation, record_key, _iso(created_at), payload),
        )
        row = connection.execute(
            "SELECT * FROM state_records WHERE operation_id=?", (operation,)
        ).fetchone()
        assert row is not None
        return self._envelope(row)

    def append(
        self,
        collection: str,
        record: dict[str, object],
        *,
        operation_id: str | None = None,
        record_key: str | None = None,
        timestamp: datetime | None = None,
    ) -> dict[str, object]:
        try:
            with self._db.transaction() as connection:
                envelope = self._append(
                    connection,
                    collection,
                    record,
                    operation_id=operation_id,
                    record_key=record_key,
                    timestamp=timestamp,
                )
        except Exception as exc:
            emit_safely(
                self.event_sink,
                StructuredEvent.create(
                    "state.append_failed",
                    "sqlite_state_repository",
                    severity="error",
                    attributes={
                        "collection": collection,
                        "error_type": type(exc).__name__,
                    },
                ),
            )
            raise
        emit_safely(
            self.event_sink,
            StructuredEvent.create(
                "state.appended",
                "sqlite_state_repository",
                entity_id=str(envelope["operation_id"]),
                attributes={"collection": collection},
            ),
        )
        return envelope

    @contextmanager
    def transaction(self) -> Iterator[SQLiteStateTransaction]:
        with self._db.transaction() as connection:
            yield SQLiteStateTransaction(self, connection)

    def records(self, collection: str) -> list[dict[str, object]]:
        _validate_name(collection, "state collection")
        with self._db.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM state_records WHERE collection=? ORDER BY sequence",
                (collection,),
            ).fetchall()
        return [self._envelope(row) for row in rows]

    def all(self, collection: str) -> list[dict[str, object]]:
        return [dict(envelope["record"]) for envelope in self.records(collection)]

    def latest_by_key(self, collection: str, record_key: str) -> dict[str, object] | None:
        _validate_name(collection, "state collection")
        _validate_name(record_key, "record key")
        with self._db.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM state_records
                WHERE collection=? AND record_key=?
                ORDER BY sequence DESC LIMIT 1
                """,
                (collection, record_key),
            ).fetchone()
        return None if row is None else dict(self._envelope(row)["record"])

    def get_latest(
        self, collection: str, key: str, value: object
    ) -> dict[str, object] | None:
        for record in reversed(self.all(collection)):
            if record.get(key) == value:
                return record
        return None


class SQLiteLockProvider:
    """Cross-process lease lock for processes sharing one SQLite database file.

    This is intentionally not called a cross-host distributed lock. PostgreSQL or a
    dedicated coordination service is required when workers do not share one host-
    local database with trustworthy file locking.
    """

    def __init__(
        self,
        path: Path,
        *,
        busy_timeout_seconds: float = 5.0,
        event_sink: EventSink | None = None,
        initialize: bool = True,
    ) -> None:
        self._db = _SQLiteDatabase(path, busy_timeout_seconds)
        self.event_sink = event_sink or NullEventSink()
        if initialize:
            self.initialize()

    @property
    def path(self) -> Path:
        return self._db.path

    def initialize(self) -> None:
        with self._db.transaction() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS infrastructure_schema_migrations (
                    component TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    applied_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS lock_leases (
                    lock_name TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    lease_token TEXT NOT NULL,
                    acquired_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            connection.execute(
                """
                INSERT INTO infrastructure_schema_migrations(component, version, applied_at)
                VALUES('lock_provider', ?, ?)
                ON CONFLICT(component) DO UPDATE SET
                    version=excluded.version, applied_at=excluded.applied_at
                """,
                (SQLITE_LOCK_SCHEMA_VERSION, _iso(_timestamp())),
            )

    def schema_version(self) -> int:
        with self._db.connect() as connection:
            row = connection.execute(
                "SELECT version FROM infrastructure_schema_migrations WHERE component='lock_provider'"
            ).fetchone()
        return 0 if row is None else int(row["version"])

    def ready(self) -> bool:
        try:
            with self._db.connect() as connection:
                connection.execute("SELECT 1 FROM lock_leases LIMIT 1").fetchone()
            return self.schema_version() == SQLITE_LOCK_SCHEMA_VERSION
        except sqlite3.Error:
            return False

    def _try_acquire(
        self, name: str, owner: str, ttl_seconds: int, timestamp: datetime
    ) -> LockLease | None:
        token = str(uuid4())
        expires = timestamp + timedelta(seconds=ttl_seconds)
        with self._db.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM lock_leases WHERE lock_name=?", (name,)
            ).fetchone()
            if row is not None and datetime.fromisoformat(row["expires_at"]) > timestamp:
                return None
            connection.execute(
                """
                INSERT INTO lock_leases(lock_name, owner_id, lease_token, acquired_at, expires_at, version)
                VALUES(?, ?, ?, ?, ?, 0)
                ON CONFLICT(lock_name) DO UPDATE SET
                    owner_id=excluded.owner_id,
                    lease_token=excluded.lease_token,
                    acquired_at=excluded.acquired_at,
                    expires_at=excluded.expires_at,
                    version=lock_leases.version+1
                """,
                (name, owner, token, _iso(timestamp), _iso(expires)),
            )
        return LockLease(name, owner, timestamp, expires, token)

    def acquire(
        self,
        name: str,
        owner: str,
        ttl_seconds: int,
        *,
        timeout_seconds: float = 0.0,
        now: datetime | None = None,
    ) -> LockLease | None:
        _validate_name(name, "lock name")
        _validate_name(owner, "lock owner")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must not be negative")
        fixed_now = _timestamp(now) if now is not None else None
        deadline = time.monotonic() + timeout_seconds
        while True:
            timestamp = fixed_now or _timestamp()
            lease = self._try_acquire(name, owner, ttl_seconds, timestamp)
            if lease is not None:
                return lease
            emit_safely(
                self.event_sink,
                StructuredEvent.create(
                    "lock.contended",
                    "sqlite_lock_provider",
                    entity_id=name,
                    severity="warning",
                    attributes={"owner": owner},
                ),
            )
            if time.monotonic() >= deadline:
                return None
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    def heartbeat(
        self,
        lease: LockLease,
        ttl_seconds: int,
        *,
        now: datetime | None = None,
    ) -> LockLease | None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if not lease.lease_token:
            return None
        timestamp = _timestamp(now)
        expires = timestamp + timedelta(seconds=ttl_seconds)
        with self._db.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE lock_leases SET expires_at=?, version=version+1
                WHERE lock_name=? AND owner_id=? AND lease_token=? AND expires_at>?
                """,
                (
                    _iso(expires),
                    lease.name,
                    lease.owner,
                    lease.lease_token,
                    _iso(timestamp),
                ),
            )
        if cursor.rowcount != 1:
            return None
        return LockLease(lease.name, lease.owner, lease.acquired_at, expires, lease.lease_token)

    def release(self, lease: LockLease) -> bool:
        if not lease.lease_token:
            return False
        with self._db.transaction() as connection:
            cursor = connection.execute(
                """
                DELETE FROM lock_leases
                WHERE lock_name=? AND owner_id=? AND lease_token=?
                """,
                (lease.name, lease.owner, lease.lease_token),
            )
        return cursor.rowcount == 1

    def assert_owned(self, lease: LockLease, *, now: datetime | None = None) -> None:
        timestamp = _timestamp(now)
        with self._db.connect() as connection:
            row = connection.execute(
                "SELECT * FROM lock_leases WHERE lock_name=?", (lease.name,)
            ).fetchone()
        if (
            row is None
            or row["owner_id"] != lease.owner
            or row["lease_token"] != lease.lease_token
            or datetime.fromisoformat(row["expires_at"]) <= timestamp
        ):
            raise LockLostError(f"lock ownership was lost: {lease.name}")

    @contextmanager
    def held(
        self,
        name: str,
        owner: str,
        ttl_seconds: int,
        *,
        timeout_seconds: float = 0.0,
    ) -> Iterator[LockLease]:
        lease = self.acquire(
            name,
            owner,
            ttl_seconds,
            timeout_seconds=timeout_seconds,
        )
        if lease is None:
            raise LockTimeoutError(f"timed out acquiring lock {name!r}")
        body_failed = False
        try:
            yield lease
        except BaseException:
            body_failed = True
            raise
        finally:
            released = self.release(lease)
            if not released and not body_failed:
                raise LockLostError(f"lock ownership was lost before release: {name}")
