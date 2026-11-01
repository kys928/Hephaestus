"""Transactional SQLite state repository and cross-process lease locks."""

from __future__ import annotations

import json
import re
import sqlite3
import time
import os
import tempfile
import threading
from contextlib import contextmanager, closing, nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4, UUID
from hephaestus.policy.storage_admission import admit_storage, reject_symlinks
from hephaestus.schemas.storage_admission import LocalStorageProfile

from hephaestus.infrastructure.observability import (
    EventSink,
    NullEventSink,
    StructuredEvent,
    emit_safely,
)

from .base import LockLease
from .durability import sync_file, sync_directory_chain


SQLITE_STATE_SCHEMA_VERSION = 1
SQLITE_LOCK_SCHEMA_VERSION = 1
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")


class StateOperationConflict(RuntimeError):
    pass


class StateIntegrityError(RuntimeError):
    """No workflow may claim healthy durable state after an integrity failure."""


class StateCommitUncertain(RuntimeError):
    """A failed acknowledgement is not proof that the SQL transaction rolled back."""
    def __init__(self, path):
        super().__init__(f"SQLite commit outcome unknown at {path}; inspect operation identities before retry")
        self.operation_ids = ()


def _schema_statements(connection, text):
    # executescript() would implicitly COMMIT the caller's BEGIN IMMEDIATE.
    for statement in text.split(';'):
        if statement.strip():
            connection.execute(statement)


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
    def __init__(self, path: Path, busy_timeout_seconds: float, read_only: bool = False,
                 storage_profile: LocalStorageProfile | None = None, *, campaign_witness=False) -> None:
        self.path = reject_symlinks(Path(path))
        self.busy_timeout_seconds = busy_timeout_seconds
        self.read_only = read_only
        self.profile = storage_profile or LocalStorageProfile()
        self.admission = None if read_only else admit_storage(self.path,self.profile)
        self._file_identity = None
        self._database_id = None
        self._pid = os.getpid()
        self._local = threading.local()
        self._witness = None
        self._require_witness = campaign_witness

    def witness(self):
        if self.read_only:
            return None
        # A second repository/queue/lock object cannot bypass a campaign's
        # previously registered fence by omitting the composition-root option.
        from hephaestus.policy.campaign_storage import campaign_control_path
        if self._witness is None and (self._require_witness or
                (campaign_control_path(self.path, self.profile) / 'commits.jsonl').exists()):
            from hephaestus.state.campaign_witness import CampaignWitness
            self._witness = CampaignWitness(self.path, self.profile)
        return self._witness

    @staticmethod
    def _identity_dict(identity):
        return None if identity is None else dict(database_id=identity[0], commit_id=identity[1])

    @staticmethod
    def _witness_required(connection):
        table = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='storage_witness'").fetchone()
        return table is not None

    def assert_witness(self, connection, witness, *, reconcile=True):
        required = self._witness_required(connection)
        if witness is None:
            if required:
                raise StateIntegrityError('campaign witness missing; restore the separately retained control journal before writing')
            return
        table = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='state_records'").fetchone()
        has_records = table is not None and connection.execute('SELECT 1 FROM state_records LIMIT 1').fetchone() is not None
        witness.assert_current(self._identity_dict(self._storage_identity(connection)),
            has_records=has_records, required=required, reconcile=reconcile)

    def _check_file_identity(self):
        if os.getpid()!=self._pid:
            raise StateIntegrityError("storage repository inherited across fork; reconstruct it in the child process")
        reject_symlinks(self.path)
        if self.path.exists():
            info=self.path.stat()
            if info.st_nlink!=1:
                raise StateIntegrityError("SQLite storage identity must not have hard-link aliases")
            actual=(info.st_dev,info.st_ino)
            if self._file_identity is not None and actual!=self._file_identity:
                raise StateIntegrityError("SQLite database file identity changed or was replaced")
        elif self._file_identity is not None:
            raise StateIntegrityError("SQLite database identity disappeared; restore explicitly before reopening")

    @staticmethod
    def _storage_identity(connection):
        if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='storage_identity'").fetchone() is None:
            return None
        row=connection.execute("SELECT database_id,commit_id FROM storage_identity WHERE singleton=1").fetchone()
        if row is None:
            raise StateIntegrityError("SQLite storage identity is incomplete")
        try:
            identity=(str(row[0]),str(row[1]))
            if any(str(UUID(value))!=value for value in identity):raise ValueError('noncanonical identity')
        except (ValueError,TypeError) as exc:
            raise StateIntegrityError("SQLite storage identity is malformed") from exc
        return identity

    @contextmanager
    def initialization_owner(self):
        if self.read_only:raise PermissionError("read-only storage cannot be initialized")
        admit_storage(self.path,self.profile)
        witness = self.witness()
        with witness.owner() if witness else nullcontext():
            if witness and not self.path.exists():
                witness.assert_current(None)
            with self._initialization_owner():
                yield

    @contextmanager
    def _initialization_owner(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        import fcntl
        descriptor=os.open(self.path.parent/f'.{self.path.name}.initialization.lock',
            os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        try:
            fcntl.flock(descriptor,fcntl.LOCK_EX)
            self._local.initializing=True
            yield
        finally:
            self._local.initializing=False
            fcntl.flock(descriptor,fcntl.LOCK_UN)
            os.close(descriptor)

    def validate_existing_component(self,component,version):
        if not self.path.exists():return
        with self.connection() as connection:
            if [str(r[0]) for r in connection.execute('PRAGMA integrity_check')]!=['ok']:
                raise StateIntegrityError("SQLite storage integrity failed before initialization; preserve originals")
            table=connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='infrastructure_schema_migrations'").fetchone()
            if table is not None:
                row=connection.execute('SELECT version FROM infrastructure_schema_migrations WHERE component=?',(component,)).fetchone()
                if row is not None and row[0]!=version:
                    raise StateIntegrityError("SQLite schema version is unsupported; no implicit upgrade or downgrade")

    def connect(self, *, read_only: bool | None = None) -> sqlite3.Connection:
        readonly = self.read_only if read_only is None else read_only
        self._check_file_identity()
        if not self.read_only:
            self.admission=admit_storage(self.path,self.profile)
        if not readonly:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.path.as_uri()+"?mode=ro" if readonly else self.path,
            timeout=self.busy_timeout_seconds,
            isolation_level=None,
            uri=readonly,
        )
        try:
            connection.row_factory = sqlite3.Row
            connection.execute(f"PRAGMA busy_timeout={int(self.busy_timeout_seconds * 1000)}")
            connection.execute("PRAGMA foreign_keys=ON")
            self._check_file_identity()
            info=self.path.stat(); self._file_identity=(info.st_dev,info.st_ino)
            identity=self._storage_identity(connection)
            if self._database_id is not None and (identity is None or identity[0]!=self._database_id):
                raise StateIntegrityError("SQLite logical database identity changed")
            if identity is not None:self._database_id=identity[0]
            if not readonly:
                if self._witness_required(connection) and self.witness() is None:
                    raise StateIntegrityError('campaign witness missing; automatic reset is forbidden')
                mode=connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
                if str(mode).lower()!='wal':raise StateIntegrityError("SQLite WAL mode was not established")
                connection.execute("PRAGMA synchronous=FULL")
                if connection.execute('PRAGMA synchronous').fetchone()[0]!=2:
                    raise StateIntegrityError("SQLite FULL synchronous durability was not established")
        except BaseException:
            connection.close()
            raise
        return connection

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        # sqlite3.Connection.__exit__ commits/rolls back; it does not close.
        # Explicit closure also keeps connections out of later worker forks.
        connection = self.connect(read_only=True)
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        if self.read_only:
            raise PermissionError("read-only state repository cannot start a write transaction")
        witness = self.witness()
        with witness.owner() if witness else nullcontext():
            if witness and not self.path.exists():
                witness.assert_current(None)
            with self._transaction_owned(witness) as connection:
                yield connection

    @contextmanager
    def _transaction_owned(self, witness):
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self.assert_witness(connection, witness)
            before = self._identity_dict(self._storage_identity(connection))
            if self._storage_identity(connection) is None and not getattr(self._local,'initializing',False):
                raise StateIntegrityError("storage identity missing; explicitly initialize a healthy admitted legacy database first")
            yield connection
            self._check_file_identity()
            identity=self._storage_identity(connection)
            if identity is not None:
                after = dict(database_id=identity[0], commit_id=str(uuid4()))
                if witness:
                    # This table makes a missing external journal fail closed,
                    # even when an ordinary low-level writer opens the backup.
                    connection.execute('CREATE TABLE IF NOT EXISTS storage_witness (singleton INTEGER PRIMARY KEY CHECK(singleton=1),version INTEGER NOT NULL)')
                    connection.execute('INSERT OR IGNORE INTO storage_witness VALUES(1,1)')
                    witness.prepare(before, after)
                connection.execute('UPDATE storage_identity SET commit_id=? WHERE singleton=1',(after['commit_id'],))
            try:
                connection.commit()
                sync_directory_chain(self.path.parent)
                self._check_file_identity()
                identity=self._storage_identity(connection)
                if identity is not None:self._database_id=identity[0]
                if witness:
                    witness.committed(before, after)
            except Exception as exc:
                raise StateCommitUncertain(self.path) from exc
        except BaseException:
            try:connection.rollback()
            except sqlite3.Error:pass
            raise
        finally:
            try:connection.close()
            except Exception as exc:raise StateCommitUncertain(self.path) from exc


class SQLiteStateTransaction:
    def __init__(self, repository: "SQLiteStateRepository", connection: sqlite3.Connection) -> None:
        self.repository = repository
        self.connection = connection
        self.operation_ids = []

    def append(
        self,
        collection: str,
        record: dict[str, object],
        *,
        operation_id: str | None = None,
        record_key: str | None = None,
        timestamp: datetime | None = None,
    ) -> dict[str, object]:
        operation_id = operation_id or str(uuid4())
        result = self.repository._append(
            self.connection,
            collection,
            record,
            operation_id=operation_id,
            record_key=record_key,
            timestamp=timestamp,
        )
        self.operation_ids.append(operation_id)
        return result


class SQLiteStateRepository:
    """Ordered JSON-safe records with idempotent operation IDs and transactions."""

    def __init__(
        self,
        path: Path,
        *,
        busy_timeout_seconds: float = 5.0,
        event_sink: EventSink | None = None,
        initialize: bool = True,
        read_only: bool = False,
        storage_profile: LocalStorageProfile | None = None,
        campaign_witness: bool = False,
    ) -> None:
        if busy_timeout_seconds < 0:
            raise ValueError("busy_timeout_seconds must not be negative")
        self._db = _SQLiteDatabase(path, busy_timeout_seconds, read_only,storage_profile,
                                   campaign_witness=campaign_witness)
        self.event_sink = event_sink or NullEventSink()
        if initialize and not read_only:
            self.initialize()

    @property
    def path(self) -> Path:
        return self._db.path

    def initialize(self) -> None:
        with self._db.initialization_owner():
            self._initialize_owned()

    def _initialize_owned(self):
        self._db.validate_existing_component('state_repository',SQLITE_STATE_SCHEMA_VERSION)
        with self._db.transaction() as connection:
            _schema_statements(connection,
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
            self._initialize_identity(connection)
            connection.execute(
                """
                INSERT INTO infrastructure_schema_migrations(component, version, applied_at)
                VALUES('state_repository', ?, ?)
                ON CONFLICT(component) DO UPDATE SET
                    version=excluded.version, applied_at=excluded.applied_at
                """,
                (SQLITE_STATE_SCHEMA_VERSION, _iso(_timestamp())),
            )

    @staticmethod
    def _initialize_identity(connection):
        connection.execute('CREATE TABLE IF NOT EXISTS storage_identity (singleton INTEGER PRIMARY KEY CHECK(singleton=1),database_id TEXT NOT NULL,commit_id TEXT NOT NULL)')
        connection.execute('INSERT OR IGNORE INTO storage_identity VALUES(1,?,?)',(str(uuid4()),str(uuid4())))

    def storage_identity(self):
        with self._db.connection() as connection:
            identity=self._db._storage_identity(connection)
        return None if identity is None else {'database_id':identity[0],'commit_id':identity[1]}

    def schema_version(self) -> int:
        with self._db.connection() as connection:
            row = connection.execute(
                "SELECT version FROM infrastructure_schema_migrations WHERE component='state_repository'"
            ).fetchone()
        return 0 if row is None else int(row["version"])

    def ready(self) -> bool:
        try:
            self.assert_integrity()
            return self.schema_version() == SQLITE_STATE_SCHEMA_VERSION
        except (sqlite3.Error, StateIntegrityError):
            return False

    def assert_integrity(self) -> None:
        try:
            with self._db.connection() as connection:
                rows = [str(row[0]) for row in connection.execute("PRAGMA integrity_check")]
            if rows != ["ok"]:
                raise StateIntegrityError("SQLite state integrity check failed; preserve the original and investigate")
        except sqlite3.Error as exc:
            raise StateIntegrityError("SQLite state cannot be verified; no automatic repair is permitted") from exc

    def snapshot_to(self, destination: Path) -> None:
        """A consistent SQLite backup, never a live main-file-only copy."""
        destination=Path(destination)
        admit_storage(destination,self._db.profile)
        if destination.exists():
            raise FileExistsError("state snapshot destination must be new")
        destination.parent.mkdir(parents=True,exist_ok=True)
        descriptor, partial = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent)
        os.close(descriptor)
        with self._db.connection() as source, closing(sqlite3.connect(partial)) as target:
            target.execute("PRAGMA synchronous=FULL")
            source.backup(target)
            if [str(row[0]) for row in target.execute("PRAGMA integrity_check")]!=["ok"]:
                raise StateIntegrityError("state snapshot is not healthy; retained for investigation")
            target.execute("PRAGMA journal_mode=DELETE")
        # Hash/read only this closed, private backup, never the live main database.
        sync_file(Path(partial))
        os.link(partial, destination, follow_symlinks=False)
        os.unlink(partial)
        sync_directory_chain(destination.parent)

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
        operation_id=operation_id or str(uuid4())
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
            if isinstance(exc,StateCommitUncertain):exc.operation_ids=(operation_id,)
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
        transaction=None
        try:
            with self._db.transaction() as connection:
                transaction=SQLiteStateTransaction(self, connection)
                yield transaction
        except StateCommitUncertain as exc:
            exc.operation_ids=tuple(transaction.operation_ids) if transaction else ()
            raise

    def records(self, collection: str) -> list[dict[str, object]]:
        _validate_name(collection, "state collection")
        with self._db.connection() as connection:
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
        with self._db.connection() as connection:
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
        _validate_name(collection,"state collection")
        with self._db.connection() as connection:
            for row in connection.execute("SELECT payload_json FROM state_records WHERE collection=? ORDER BY sequence DESC",(collection,)):
                record=json.loads(row[0])
                if record.get(key)==value:
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
        storage_profile: LocalStorageProfile | None = None,
    ) -> None:
        self._db = _SQLiteDatabase(path, busy_timeout_seconds, storage_profile=storage_profile)
        self.event_sink = event_sink or NullEventSink()
        if initialize:
            self.initialize()

    @property
    def path(self) -> Path:
        return self._db.path

    def initialize(self) -> None:
        with self._db.initialization_owner():
            self._initialize_owned()

    def _initialize_owned(self):
        self._db.validate_existing_component('lock_provider',SQLITE_LOCK_SCHEMA_VERSION)
        with self._db.transaction() as connection:
            _schema_statements(connection,
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
            SQLiteStateRepository._initialize_identity(connection)
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
        with self._db.connection() as connection:
            row = connection.execute(
                "SELECT version FROM infrastructure_schema_migrations WHERE component='lock_provider'"
            ).fetchone()
        return 0 if row is None else int(row["version"])

    def ready(self) -> bool:
        try:
            with self._db.connection() as connection:
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
        with self._db.connection() as connection:
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
