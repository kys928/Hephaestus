"""Transactional SQLite job queue for durable, multi-process local execution."""

from __future__ import annotations

import json
import sqlite3
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

from .models import JobRecord, JobStatus, TERMINAL_JOB_STATUSES
from .queue import (
    IdempotencyConflict,
    InvalidJobTransition,
    JobNotFound,
    LeaseOwnershipError,
    _timestamp,
)


SQLITE_QUEUE_SCHEMA_VERSION = 1
_ACTIVE = {JobStatus.LEASED, JobStatus.RUNNING}


def _utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return None if value is None else _utc(value).isoformat()


def _datetime(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


def _valid_payload_reference(value: str) -> bool:
    return bool(value.strip()) and len(value) <= 2048 and not any(ord(char) < 32 for char in value)


class SQLiteJobQueue:
    """Durable queue with SQLite write transactions and lease fencing.

    SQLite serializes writers and is safe for competing processes that can access the
    same database file. It is not a cross-host consensus system and should not be put
    on a network filesystem whose locking guarantees are unknown.
    """

    def __init__(
        self,
        path: Path,
        *,
        maximum_attempts: int = 3,
        maximum_lease_expirations: int = 3,
        busy_timeout_seconds: float = 5.0,
        event_sink: EventSink | None = None,
        initialize: bool = True,
    ) -> None:
        if maximum_attempts <= 0:
            raise ValueError("maximum_attempts must be positive")
        if maximum_lease_expirations <= 0:
            raise ValueError("maximum_lease_expirations must be positive")
        if busy_timeout_seconds < 0:
            raise ValueError("busy_timeout_seconds must not be negative")
        self.path = Path(path)
        self.maximum_attempts = maximum_attempts
        self.maximum_lease_expirations = maximum_lease_expirations
        self.busy_timeout_seconds = busy_timeout_seconds
        self._events = event_sink or NullEventSink()
        if initialize:
            self.initialize()

    def _connect(self) -> sqlite3.Connection:
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
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._transaction() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS infrastructure_schema_migrations (
                    component TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    applied_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    payload_ref TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    run_id TEXT,
                    experiment_id TEXT,
                    idempotency_key TEXT UNIQUE,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    lease_owner TEXT,
                    lease_token TEXT,
                    lease_expires_at TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    result_ref TEXT,
                    error_ref TEXT,
                    cancellation_requested INTEGER NOT NULL DEFAULT 0 CHECK (cancellation_requested IN (0, 1)),
                    cancellation_acknowledged INTEGER NOT NULL DEFAULT 0 CHECK (cancellation_acknowledged IN (0, 1)),
                    lease_expiration_count INTEGER NOT NULL DEFAULT 0 CHECK (lease_expiration_count >= 0),
                    maximum_attempts INTEGER NOT NULL CHECK (maximum_attempts > 0),
                    dead_letter_reason TEXT,
                    dead_lettered_at TEXT,
                    replay_count INTEGER NOT NULL DEFAULT 0 CHECK (replay_count >= 0),
                    version INTEGER NOT NULL DEFAULT 0 CHECK (version >= 0)
                );
                CREATE INDEX IF NOT EXISTS jobs_lease_order
                    ON jobs(status, created_at, job_id);
                CREATE INDEX IF NOT EXISTS jobs_lease_expiry
                    ON jobs(status, lease_expires_at);
                CREATE TABLE IF NOT EXISTS job_audit (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    event_type TEXT NOT NULL,
                    actor TEXT,
                    reason TEXT,
                    evidence_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            now = _iso(datetime.now(timezone.utc))
            connection.execute(
                """
                INSERT INTO infrastructure_schema_migrations(component, version, applied_at)
                VALUES('job_queue', ?, ?)
                ON CONFLICT(component) DO UPDATE SET
                    version=excluded.version,
                    applied_at=excluded.applied_at
                """,
                (SQLITE_QUEUE_SCHEMA_VERSION, now),
            )

    def schema_version(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT version FROM infrastructure_schema_migrations WHERE component='job_queue'"
            ).fetchone()
        return 0 if row is None else int(row["version"])

    def ready(self) -> bool:
        try:
            with self._connect() as connection:
                connection.execute("SELECT 1 FROM jobs LIMIT 1").fetchone()
            return self.schema_version() == SQLITE_QUEUE_SCHEMA_VERSION
        except sqlite3.Error:
            return False

    @staticmethod
    def _record(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=str(row["job_id"]),
            payload_ref=str(row["payload_ref"]),
            owner_id=str(row["owner_id"]),
            run_id=row["run_id"],
            experiment_id=row["experiment_id"],
            idempotency_key=row["idempotency_key"],
            status=JobStatus(str(row["status"])),
            attempt_count=int(row["attempt_count"]),
            created_at=_datetime(row["created_at"]),  # type: ignore[arg-type]
            updated_at=_datetime(row["updated_at"]),  # type: ignore[arg-type]
            lease_owner=row["lease_owner"],
            lease_expires_at=_datetime(row["lease_expires_at"]),
            started_at=_datetime(row["started_at"]),
            finished_at=_datetime(row["finished_at"]),
            result_ref=row["result_ref"],
            error_ref=row["error_ref"],
            cancellation_requested=bool(row["cancellation_requested"]),
            cancellation_acknowledged=bool(row["cancellation_acknowledged"]),
            lease_token=row["lease_token"],
            lease_expiration_count=int(row["lease_expiration_count"]),
            maximum_attempts=int(row["maximum_attempts"]),
            dead_letter_reason=row["dead_letter_reason"],
            dead_lettered_at=_datetime(row["dead_lettered_at"]),
            replay_count=int(row["replay_count"]),
        )

    @classmethod
    def _required(cls, connection: sqlite3.Connection, job_id: str) -> JobRecord:
        row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise JobNotFound(job_id)
        return cls._record(row)

    @staticmethod
    def _audit(
        connection: sqlite3.Connection,
        job_id: str,
        event_type: str,
        timestamp: datetime,
        *,
        actor: str | None = None,
        reason: str | None = None,
        evidence: dict[str, object] | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO job_audit(job_id, event_type, actor, reason, evidence_json, created_at)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                event_type,
                actor,
                reason,
                json.dumps(evidence or {}, sort_keys=True, separators=(",", ":")),
                _iso(timestamp),
            ),
        )

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
        emit_safely(
            self._events,
            StructuredEvent.create(
                event_type,
                "sqlite_job_queue",
                entity_id=record.job_id,
                severity=severity,
                attributes=details,
                timestamp=record.updated_at,
            ),
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
        maximum_attempts: int | None = None,
        now: datetime | None = None,
    ) -> JobRecord:
        timestamp = _timestamp(now)
        max_attempts = maximum_attempts or self.maximum_attempts
        if max_attempts <= 0:
            raise ValueError("maximum_attempts must be positive")
        resolved_job_id = job_id or str(uuid4())
        with self._transaction() as connection:
            if idempotency_key:
                existing_row = connection.execute(
                    "SELECT * FROM jobs WHERE idempotency_key=?", (idempotency_key,)
                ).fetchone()
                if existing_row is not None:
                    existing = self._record(existing_row)
                    if (
                        existing.payload_ref,
                        existing.owner_id,
                        existing.run_id,
                        existing.experiment_id,
                    ) != (payload_ref, owner_id, run_id, experiment_id):
                        raise IdempotencyConflict(
                            f"idempotency key {idempotency_key!r} was already used for another job"
                        )
                    return existing
            try:
                connection.execute(
                    """
                    INSERT INTO jobs(
                        job_id, payload_ref, owner_id, run_id, experiment_id,
                        idempotency_key, status, attempt_count, created_at,
                        updated_at, maximum_attempts
                    ) VALUES(?, ?, ?, ?, ?, ?, 'queued', 0, ?, ?, ?)
                    """,
                    (
                        resolved_job_id,
                        payload_ref,
                        owner_id,
                        run_id,
                        experiment_id,
                        idempotency_key,
                        _iso(timestamp),
                        _iso(timestamp),
                        max_attempts,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise IdempotencyConflict(f"job identity already exists: {resolved_job_id}") from exc
            self._audit(connection, resolved_job_id, "job.queued", timestamp, actor=owner_id)
            record = self._required(connection, resolved_job_id)
        self._emit("job.queued", record)
        return record

    def get(self, job_id: str) -> JobRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return None if row is None else self._record(row)

    def all(self) -> list[JobRecord]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM jobs ORDER BY created_at, job_id").fetchall()
        return [self._record(row) for row in rows]

    def _dead_letter_tx(
        self,
        connection: sqlite3.Connection,
        record: JobRecord,
        reason: str,
        timestamp: datetime,
        *,
        actor: str | None = None,
        error_ref: str | None = None,
    ) -> JobRecord:
        connection.execute(
            """
            UPDATE jobs SET
                status='dead_letter', updated_at=?, finished_at=?, lease_owner=NULL,
                lease_token=NULL, lease_expires_at=NULL, dead_letter_reason=?,
                dead_lettered_at=?, error_ref=COALESCE(?, error_ref), version=version+1
            WHERE job_id=?
            """,
            (_iso(timestamp), _iso(timestamp), reason, _iso(timestamp), error_ref, record.job_id),
        )
        self._audit(
            connection,
            record.job_id,
            "job.dead_lettered",
            timestamp,
            actor=actor,
            reason=reason,
            evidence={"previous_status": record.status.value, "error_ref": error_ref},
        )
        return self._required(connection, record.job_id)

    def _recover_expired_tx(
        self, connection: sqlite3.Connection, timestamp: datetime
    ) -> list[JobRecord]:
        rows = connection.execute(
            """
            SELECT * FROM jobs
            WHERE status IN ('leased', 'running') AND lease_expires_at <= ?
            ORDER BY lease_expires_at, job_id
            """,
            (_iso(timestamp),),
        ).fetchall()
        recovered: list[JobRecord] = []
        for row in rows:
            record = self._record(row)
            expirations = record.lease_expiration_count + 1
            if record.cancellation_requested:
                connection.execute(
                    """
                    UPDATE jobs SET status='cancelled', updated_at=?, finished_at=?,
                        lease_owner=NULL, lease_token=NULL, lease_expires_at=NULL,
                        cancellation_acknowledged=1, lease_expiration_count=?, version=version+1
                    WHERE job_id=?
                    """,
                    (_iso(timestamp), _iso(timestamp), expirations, record.job_id),
                )
                self._audit(
                    connection,
                    record.job_id,
                    "job.cancelled",
                    timestamp,
                    reason="lease_expired_after_cancellation_request",
                )
                recovered.append(self._required(connection, record.job_id))
            elif (
                record.attempt_count >= record.maximum_attempts
                or expirations >= self.maximum_lease_expirations
            ):
                connection.execute(
                    "UPDATE jobs SET lease_expiration_count=? WHERE job_id=?",
                    (expirations, record.job_id),
                )
                refreshed = self._required(connection, record.job_id)
                recovered.append(
                    self._dead_letter_tx(
                        connection,
                        refreshed,
                        "lease_expiration_limit_exceeded",
                        timestamp,
                        error_ref="lease_expired",
                    )
                )
            else:
                connection.execute(
                    """
                    UPDATE jobs SET status='queued', updated_at=?, lease_owner=NULL,
                        lease_token=NULL, lease_expires_at=NULL, started_at=NULL,
                        finished_at=NULL, error_ref='lease_expired',
                        lease_expiration_count=?, version=version+1
                    WHERE job_id=?
                    """,
                    (_iso(timestamp), expirations, record.job_id),
                )
                self._audit(
                    connection,
                    record.job_id,
                    "job.lease_recovered",
                    timestamp,
                    reason="lease_expired",
                    evidence={"previous_owner": record.lease_owner},
                )
                recovered.append(self._required(connection, record.job_id))
        return recovered

    def recover_expired_leases(self, *, now: datetime | None = None) -> list[JobRecord]:
        timestamp = _timestamp(now)
        with self._transaction() as connection:
            recovered = self._recover_expired_tx(connection, timestamp)
        for record in recovered:
            event = "job.dead_lettered" if record.status is JobStatus.DEAD_LETTER else "job.lease_recovered"
            self._emit(event, record, severity="warning")
        return recovered

    def lease_next(
        self, worker_id: str, lease_seconds: int, *, now: datetime | None = None
    ) -> JobRecord | None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if not worker_id.strip():
            raise ValueError("worker_id must not be empty")
        timestamp = _timestamp(now)
        leased: JobRecord | None = None
        dead_letters: list[JobRecord] = []
        recovered: list[JobRecord] = []
        with self._transaction() as connection:
            recovered = self._recover_expired_tx(connection, timestamp)
            while True:
                row = connection.execute(
                    """
                    SELECT * FROM jobs
                    WHERE status='queued' AND cancellation_requested=0
                    ORDER BY created_at, job_id LIMIT 1
                    """
                ).fetchone()
                if row is None:
                    break
                candidate = self._record(row)
                if not _valid_payload_reference(candidate.payload_ref):
                    dead_letters.append(
                        self._dead_letter_tx(
                            connection,
                            candidate,
                            "malformed_payload_reference",
                            timestamp,
                            error_ref="malformed_payload_reference",
                        )
                    )
                    continue
                if candidate.attempt_count >= candidate.maximum_attempts:
                    dead_letters.append(
                        self._dead_letter_tx(
                            connection,
                            candidate,
                            "maximum_attempts_exceeded",
                            timestamp,
                        )
                    )
                    continue
                token = str(uuid4())
                cursor = connection.execute(
                    """
                    UPDATE jobs SET status='leased', attempt_count=attempt_count+1,
                        lease_owner=?, lease_token=?, lease_expires_at=?, updated_at=?,
                        version=version+1
                    WHERE job_id=? AND status='queued' AND version=?
                    """,
                    (
                        worker_id,
                        token,
                        _iso(timestamp + timedelta(seconds=lease_seconds)),
                        _iso(timestamp),
                        candidate.job_id,
                        int(row["version"]),
                    ),
                )
                if cursor.rowcount != 1:
                    continue
                self._audit(
                    connection,
                    candidate.job_id,
                    "job.leased",
                    timestamp,
                    actor=worker_id,
                    evidence={"lease_token": token, "attempt": candidate.attempt_count + 1},
                )
                leased = self._required(connection, candidate.job_id)
                break
        for record in dead_letters:
            self._emit("job.dead_lettered", record, severity="error")
        for record in recovered:
            event = (
                "job.dead_lettered"
                if record.status is JobStatus.DEAD_LETTER
                else "job.cancelled"
                if record.status is JobStatus.CANCELLED
                else "job.lease_recovered"
            )
            self._emit(event, record, severity="warning")
        if leased is not None:
            self._emit(
                "job.leased",
                leased,
                attributes={
                    "worker_id": worker_id,
                    "queue_delay_seconds": max(
                        0.0, (timestamp - leased.created_at).total_seconds()
                    ),
                },
            )
        return leased

    @staticmethod
    def _assert_fenced_owner(
        record: JobRecord, worker_id: str, lease_token: str | None, timestamp: datetime
    ) -> None:
        if not lease_token or record.lease_owner != worker_id or record.lease_token != lease_token:
            raise LeaseOwnershipError(
                f"job {record.job_id} lease fencing token or owner does not match"
            )
        if record.lease_expires_at is None or record.lease_expires_at <= timestamp:
            raise LeaseOwnershipError(f"job {record.job_id} lease has expired")

    def start(
        self,
        job_id: str,
        worker_id: str,
        *,
        lease_token: str | None = None,
        now: datetime | None = None,
    ) -> JobRecord:
        timestamp = _timestamp(now)
        with self._transaction() as connection:
            record = self._required(connection, job_id)
            self._assert_fenced_owner(record, worker_id, lease_token, timestamp)
            if record.status is not JobStatus.LEASED:
                raise InvalidJobTransition(f"cannot start {record.status.value} job {job_id}")
            if record.cancellation_requested:
                updated = self._acknowledge_tx(connection, record, worker_id, timestamp)
                event_type = "job.cancelled"
            else:
                connection.execute(
                    """
                    UPDATE jobs SET status='running', started_at=COALESCE(started_at, ?),
                        updated_at=?, version=version+1 WHERE job_id=?
                    """,
                    (_iso(timestamp), _iso(timestamp), job_id),
                )
                self._audit(connection, job_id, "job.running", timestamp, actor=worker_id)
                updated = self._required(connection, job_id)
                event_type = "job.running"
        self._emit(event_type, updated, attributes={"worker_id": worker_id})
        return updated

    def heartbeat(
        self,
        job_id: str,
        worker_id: str,
        lease_seconds: int,
        *,
        lease_token: str | None = None,
        now: datetime | None = None,
    ) -> JobRecord:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        timestamp = _timestamp(now)
        with self._transaction() as connection:
            record = self._required(connection, job_id)
            self._assert_fenced_owner(record, worker_id, lease_token, timestamp)
            if record.status not in _ACTIVE:
                raise InvalidJobTransition(
                    f"cannot heartbeat {record.status.value} job {job_id}"
                )
            connection.execute(
                """
                UPDATE jobs SET lease_expires_at=?, updated_at=?, version=version+1
                WHERE job_id=?
                """,
                (_iso(timestamp + timedelta(seconds=lease_seconds)), _iso(timestamp), job_id),
            )
            self._audit(connection, job_id, "worker.heartbeat", timestamp, actor=worker_id)
            updated = self._required(connection, job_id)
        self._emit("worker.heartbeat", updated, attributes={"worker_id": worker_id})
        return updated

    def request_cancellation(
        self, job_id: str, *, now: datetime | None = None
    ) -> JobRecord:
        timestamp = _timestamp(now)
        with self._transaction() as connection:
            record = self._required(connection, job_id)
            if record.status in TERMINAL_JOB_STATUSES:
                return record
            if record.status is JobStatus.QUEUED:
                connection.execute(
                    """
                    UPDATE jobs SET status='cancelled', cancellation_requested=1,
                        cancellation_acknowledged=1, updated_at=?, finished_at=?,
                        version=version+1 WHERE job_id=?
                    """,
                    (_iso(timestamp), _iso(timestamp), job_id),
                )
                event_type = "job.cancelled"
            else:
                connection.execute(
                    """
                    UPDATE jobs SET cancellation_requested=1, updated_at=?,
                        version=version+1 WHERE job_id=?
                    """,
                    (_iso(timestamp), job_id),
                )
                event_type = "job.cancellation_requested"
            self._audit(connection, job_id, event_type, timestamp)
            updated = self._required(connection, job_id)
        self._emit(event_type, updated, severity="warning" if event_type.endswith("requested") else "info")
        return updated

    def _acknowledge_tx(
        self,
        connection: sqlite3.Connection,
        record: JobRecord,
        worker_id: str,
        timestamp: datetime,
    ) -> JobRecord:
        if not record.cancellation_requested:
            raise InvalidJobTransition(f"job {record.job_id} has no cancellation request")
        if record.status not in _ACTIVE:
            raise InvalidJobTransition(
                f"cannot cancel {record.status.value} job {record.job_id}"
            )
        connection.execute(
            """
            UPDATE jobs SET status='cancelled', cancellation_acknowledged=1,
                updated_at=?, finished_at=?, lease_owner=NULL, lease_token=NULL,
                lease_expires_at=NULL, version=version+1 WHERE job_id=?
            """,
            (_iso(timestamp), _iso(timestamp), record.job_id),
        )
        self._audit(
            connection, record.job_id, "job.cancelled", timestamp, actor=worker_id
        )
        return self._required(connection, record.job_id)

    def acknowledge_cancellation(
        self,
        job_id: str,
        worker_id: str,
        *,
        lease_token: str | None = None,
        now: datetime | None = None,
    ) -> JobRecord:
        timestamp = _timestamp(now)
        with self._transaction() as connection:
            record = self._required(connection, job_id)
            self._assert_fenced_owner(record, worker_id, lease_token, timestamp)
            updated = self._acknowledge_tx(connection, record, worker_id, timestamp)
        self._emit("job.cancelled", updated)
        return updated

    def complete(
        self,
        job_id: str,
        worker_id: str,
        result_ref: str | None,
        *,
        lease_token: str | None = None,
        now: datetime | None = None,
    ) -> JobRecord:
        timestamp = _timestamp(now)
        with self._transaction() as connection:
            record = self._required(connection, job_id)
            self._assert_fenced_owner(record, worker_id, lease_token, timestamp)
            if record.status is not JobStatus.RUNNING:
                raise InvalidJobTransition(
                    f"cannot complete {record.status.value} job {job_id}"
                )
            if record.cancellation_requested:
                updated = self._acknowledge_tx(connection, record, worker_id, timestamp)
                event_type = "job.cancelled"
            else:
                connection.execute(
                    """
                    UPDATE jobs SET status='succeeded', result_ref=?, updated_at=?,
                        finished_at=?, lease_owner=NULL, lease_token=NULL,
                        lease_expires_at=NULL, version=version+1 WHERE job_id=?
                    """,
                    (result_ref, _iso(timestamp), _iso(timestamp), job_id),
                )
                self._audit(connection, job_id, "job.succeeded", timestamp, actor=worker_id)
                updated = self._required(connection, job_id)
                event_type = "job.succeeded"
        self._emit(event_type, updated)
        return updated

    def fail(
        self,
        job_id: str,
        worker_id: str,
        error_ref: str,
        *,
        retryable: bool = True,
        lease_token: str | None = None,
        now: datetime | None = None,
    ) -> JobRecord:
        timestamp = _timestamp(now)
        with self._transaction() as connection:
            record = self._required(connection, job_id)
            self._assert_fenced_owner(record, worker_id, lease_token, timestamp)
            if record.status is not JobStatus.RUNNING:
                raise InvalidJobTransition(f"cannot fail {record.status.value} job {job_id}")
            if record.cancellation_requested:
                updated = self._acknowledge_tx(connection, record, worker_id, timestamp)
                event_type = "job.cancelled"
            elif not retryable:
                updated = self._dead_letter_tx(
                    connection,
                    record,
                    "non_retryable_failure",
                    timestamp,
                    actor=worker_id,
                    error_ref=error_ref,
                )
                event_type = "job.dead_lettered"
            elif record.attempt_count >= record.maximum_attempts:
                updated = self._dead_letter_tx(
                    connection,
                    record,
                    "maximum_attempts_exceeded",
                    timestamp,
                    actor=worker_id,
                    error_ref=error_ref,
                )
                event_type = "job.dead_lettered"
            else:
                connection.execute(
                    """
                    UPDATE jobs SET status='failed', error_ref=?, updated_at=?,
                        finished_at=?, lease_owner=NULL, lease_token=NULL,
                        lease_expires_at=NULL, version=version+1 WHERE job_id=?
                    """,
                    (error_ref, _iso(timestamp), _iso(timestamp), job_id),
                )
                self._audit(
                    connection,
                    job_id,
                    "job.failed",
                    timestamp,
                    actor=worker_id,
                    evidence={"retryable": True, "error_ref": error_ref},
                )
                updated = self._required(connection, job_id)
                event_type = "job.failed"
        self._emit(event_type, updated, severity="error")
        return updated

    def retry(self, job_id: str, *, now: datetime | None = None) -> JobRecord:
        timestamp = _timestamp(now)
        with self._transaction() as connection:
            record = self._required(connection, job_id)
            if record.status not in {JobStatus.FAILED, JobStatus.EXPIRED}:
                raise InvalidJobTransition(
                    f"cannot retry {record.status.value} job {job_id}"
                )
            if record.attempt_count >= record.maximum_attempts:
                updated = self._dead_letter_tx(
                    connection, record, "maximum_attempts_exceeded", timestamp
                )
                event_type = "job.dead_lettered"
            else:
                connection.execute(
                    """
                    UPDATE jobs SET status='queued', updated_at=?, started_at=NULL,
                        finished_at=NULL, result_ref=NULL, lease_owner=NULL,
                        lease_token=NULL, lease_expires_at=NULL,
                        error_ref=NULL,
                        cancellation_requested=0, cancellation_acknowledged=0,
                        version=version+1 WHERE job_id=?
                    """,
                    (_iso(timestamp), job_id),
                )
                self._audit(connection, job_id, "job.retry_queued", timestamp)
                updated = self._required(connection, job_id)
                event_type = "job.retry_queued"
        self._emit(event_type, updated, severity="warning")
        return updated

    def replay_dead_letter(
        self,
        job_id: str,
        *,
        actor: str,
        reason: str,
        now: datetime | None = None,
    ) -> JobRecord:
        if not actor.strip() or not reason.strip():
            raise ValueError("dead-letter replay requires an actor and reason")
        timestamp = _timestamp(now)
        with self._transaction() as connection:
            record = self._required(connection, job_id)
            if record.status is not JobStatus.DEAD_LETTER:
                raise InvalidJobTransition(
                    f"cannot replay {record.status.value} job {job_id}"
                )
            self._audit(
                connection,
                job_id,
                "job.dead_letter_replayed",
                timestamp,
                actor=actor,
                reason=reason,
                evidence={
                    "dead_letter_reason": record.dead_letter_reason,
                    "error_ref": record.error_ref,
                    "attempt_count": record.attempt_count,
                },
            )
            connection.execute(
                """
                UPDATE jobs SET status='queued', updated_at=?, started_at=NULL,
                    finished_at=NULL, lease_owner=NULL, lease_token=NULL,
                    lease_expires_at=NULL, cancellation_requested=0,
                    cancellation_acknowledged=0, dead_letter_reason=NULL,
                    dead_lettered_at=NULL, replay_count=replay_count+1,
                    attempt_count=0, lease_expiration_count=0, error_ref=NULL,
                    version=version+1 WHERE job_id=?
                """,
                (_iso(timestamp), job_id),
            )
            updated = self._required(connection, job_id)
        self._emit(
            "job.dead_letter_replayed",
            updated,
            severity="warning",
            attributes={"actor": actor, "reason": reason},
        )
        return updated

    def audit_records(self, job_id: str) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM job_audit WHERE job_id=? ORDER BY sequence", (job_id,)
            ).fetchall()
        return [
            {
                "sequence": int(row["sequence"]),
                "job_id": str(row["job_id"]),
                "event_type": str(row["event_type"]),
                "actor": row["actor"],
                "reason": row["reason"],
                "evidence": json.loads(row["evidence_json"]),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]
