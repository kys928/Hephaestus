"""Optional PostgreSQL cross-host lease-lock adapter."""

from __future__ import annotations

import re
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterator
from uuid import uuid4

from hephaestus.infrastructure.capabilities import OptionalCapabilityError

from .base import LockLease
from .sqlite import LockLostError, LockTimeoutError


_LOCK_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")


def _now(value: datetime | None = None) -> datetime:
    resolved = value or datetime.now(timezone.utc)
    if resolved.tzinfo is None or resolved.utcoffset() is None:
        raise ValueError("lock timestamps must be timezone-aware")
    return resolved.astimezone(timezone.utc)


class PostgresLeaseLockProvider:
    """Database lease rows with cross-host fencing through PostgreSQL transactions."""

    def __init__(self, connection_factory: Callable[[], Any], *, initialize: bool = False) -> None:
        self.connection_factory = connection_factory
        if initialize:
            self.initialize()

    @classmethod
    def from_psycopg(cls, dsn: str, *, initialize: bool = False) -> "PostgresLeaseLockProvider":
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise OptionalCapabilityError(
                "PostgreSQL locking requires the 'postgres' optional dependencies"
            ) from exc
        return cls(lambda: psycopg.connect(dsn), initialize=initialize)

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        connection = self.connection_factory()
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS infrastructure_lock_leases (
                    lock_name TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    lease_token UUID NOT NULL,
                    acquired_at TIMESTAMPTZ NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL,
                    version BIGINT NOT NULL DEFAULT 0
                )
                """
            )

    def ready(self) -> bool:
        try:
            with self._connection() as connection, connection.cursor() as cursor:
                cursor.execute("SELECT 1 FROM infrastructure_lock_leases LIMIT 1")
            return True
        except Exception:
            return False

    @staticmethod
    def _validate(name: str, owner: str, ttl_seconds: int) -> None:
        if not _LOCK_NAME.fullmatch(name) or not _LOCK_NAME.fullmatch(owner):
            raise ValueError("lock name and owner must use bounded safe identifiers")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")

    def _try_acquire(
        self, name: str, owner: str, ttl_seconds: int, timestamp: datetime
    ) -> LockLease | None:
        token = str(uuid4())
        expires = timestamp + timedelta(seconds=ttl_seconds)
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO infrastructure_lock_leases(
                    lock_name, owner_id, lease_token, acquired_at, expires_at, version
                ) VALUES(%s, %s, %s, %s, %s, 0)
                ON CONFLICT(lock_name) DO UPDATE SET
                    owner_id=EXCLUDED.owner_id,
                    lease_token=EXCLUDED.lease_token,
                    acquired_at=EXCLUDED.acquired_at,
                    expires_at=EXCLUDED.expires_at,
                    version=infrastructure_lock_leases.version+1
                WHERE infrastructure_lock_leases.expires_at <= %s
                RETURNING lock_name
                """,
                (name, owner, token, timestamp, expires, timestamp),
            )
            acquired = cursor.fetchone()
        return None if acquired is None else LockLease(name, owner, timestamp, expires, token)

    def acquire(
        self,
        name: str,
        owner: str,
        ttl_seconds: int,
        *,
        timeout_seconds: float = 0.0,
        now: datetime | None = None,
    ) -> LockLease | None:
        self._validate(name, owner, ttl_seconds)
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must not be negative")
        fixed = _now(now) if now is not None else None
        deadline = time.monotonic() + timeout_seconds
        while True:
            lease = self._try_acquire(name, owner, ttl_seconds, fixed or _now())
            if lease is not None:
                return lease
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
        self._validate(lease.name, lease.owner, ttl_seconds)
        if not lease.lease_token:
            return None
        timestamp = _now(now)
        expires = timestamp + timedelta(seconds=ttl_seconds)
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE infrastructure_lock_leases
                SET expires_at=%s, version=version+1
                WHERE lock_name=%s AND owner_id=%s AND lease_token=%s
                  AND expires_at>%s
                RETURNING lock_name
                """,
                (expires, lease.name, lease.owner, lease.lease_token, timestamp),
            )
            renewed = cursor.fetchone()
        if renewed is None:
            return None
        return LockLease(lease.name, lease.owner, lease.acquired_at, expires, lease.lease_token)

    def release(self, lease: LockLease) -> bool:
        if not lease.lease_token:
            return False
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM infrastructure_lock_leases
                WHERE lock_name=%s AND owner_id=%s AND lease_token=%s
                """,
                (lease.name, lease.owner, lease.lease_token),
            )
            deleted = cursor.rowcount
        return deleted == 1

    def assert_owned(self, lease: LockLease) -> None:
        if not lease.lease_token:
            raise LockLostError(f"lock ownership was lost: {lease.name}")
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT 1 FROM infrastructure_lock_leases
                WHERE lock_name=%s AND owner_id=%s AND lease_token=%s
                  AND expires_at>CURRENT_TIMESTAMP
                """,
                (lease.name, lease.owner, lease.lease_token),
            )
            owned = cursor.fetchone()
        if owned is None:
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
