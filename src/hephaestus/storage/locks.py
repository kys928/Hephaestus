"""Lock adapter with a process-local reference implementation."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

from .base import LockLease


def _now(value: datetime | None) -> datetime:
    resolved = value or datetime.now(timezone.utc)
    if resolved.tzinfo is None or resolved.utcoffset() is None:
        raise ValueError("lock timestamps must be timezone-aware")
    return resolved


class InMemoryLockProvider:
    """Duplicate-owner protection inside one process; not a distributed lock."""

    def __init__(self) -> None:
        self._leases: dict[str, LockLease] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _validate_ttl(ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")

    def acquire(
        self, name: str, owner: str, ttl_seconds: int, *, now: datetime | None = None
    ) -> LockLease | None:
        self._validate_ttl(ttl_seconds)
        timestamp = _now(now)
        with self._lock:
            existing = self._leases.get(name)
            if existing is not None and existing.expires_at > timestamp:
                return None
            lease = LockLease(name, owner, timestamp, timestamp + timedelta(seconds=ttl_seconds))
            self._leases[name] = lease
            return lease

    def heartbeat(
        self, lease: LockLease, ttl_seconds: int, *, now: datetime | None = None
    ) -> LockLease | None:
        self._validate_ttl(ttl_seconds)
        timestamp = _now(now)
        with self._lock:
            existing = self._leases.get(lease.name)
            if existing != lease or existing.expires_at <= timestamp:
                return None
            renewed = LockLease(
                existing.name,
                existing.owner,
                existing.acquired_at,
                timestamp + timedelta(seconds=ttl_seconds),
            )
            self._leases[lease.name] = renewed
            return renewed

    def release(self, lease: LockLease) -> bool:
        with self._lock:
            if self._leases.get(lease.name) != lease:
                return False
            del self._leases[lease.name]
            return True
