"""Storage, append-repository, and lock adapter protocols."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Protocol


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    artifact_ref: str
    content_hash: str
    hash_algorithm: str
    byte_size: int
    storage_path: str
    created_at: datetime
    media_type: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        return payload


class ArtifactStore(Protocol):
    def put_bytes(
        self,
        data: bytes,
        *,
        expected_hash: str | None = None,
        media_type: str | None = None,
    ) -> ArtifactRecord: ...

    def put_file(
        self,
        source: Path,
        *,
        expected_hash: str | None = None,
        media_type: str | None = None,
    ) -> ArtifactRecord: ...

    def open(self, artifact_ref: str) -> BinaryIO: ...

    def verify(self, artifact_ref: str) -> bool: ...


class StateRepository(Protocol):
    def append(self, collection: str, record: dict[str, object]) -> None: ...

    def all(self, collection: str) -> list[dict[str, object]]: ...

    def get_latest(
        self, collection: str, key: str, value: object
    ) -> dict[str, object] | None: ...


@dataclass(frozen=True, slots=True)
class LockLease:
    name: str
    owner: str
    acquired_at: datetime
    expires_at: datetime
    lease_token: str | None = None


class DistributedLockProvider(Protocol):
    """Adapter point only; implementations must document their actual scope."""

    def acquire(
        self, name: str, owner: str, ttl_seconds: int, *, now: datetime | None = None
    ) -> LockLease | None: ...

    def heartbeat(
        self, lease: LockLease, ttl_seconds: int, *, now: datetime | None = None
    ) -> LockLease | None: ...

    def release(self, lease: LockLease) -> bool: ...
