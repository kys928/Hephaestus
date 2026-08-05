"""Content-addressed artifact and JSONL state adapters for local filesystems."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

from hephaestus.infrastructure.observability import EventSink, NullEventSink, StructuredEvent

from .base import ArtifactRecord

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has thread-only semantics here
    fcntl = None  # type: ignore[assignment]


class ArtifactIntegrityError(ValueError):
    pass


_ARTIFACT_REF = re.compile(r"^sha256:([0-9a-f]{64})$")
_COLLECTION = re.compile(r"^[A-Za-z0-9_.-]+$")


def _normalized_expected_hash(expected_hash: str | None) -> str | None:
    if expected_hash is None:
        return None
    return expected_hash.removeprefix("sha256:").lower()


@dataclass(slots=True)
class FileSystemArtifactStore:
    root: Path
    event_sink: EventSink = field(default_factory=NullEventSink)

    def _path_for_digest(self, digest: str) -> Path:
        return self.root / "objects" / "sha256" / digest[:2] / digest

    @staticmethod
    def _digest_from_ref(artifact_ref: str) -> str:
        match = _ARTIFACT_REF.fullmatch(artifact_ref)
        if not match:
            raise ValueError(f"invalid immutable artifact reference: {artifact_ref!r}")
        return match.group(1)

    def _failure(self, operation: str, exc: Exception) -> None:
        self.event_sink.emit(
            StructuredEvent.create(
                "storage.failure",
                "filesystem_artifact_store",
                severity="error",
                attributes={"operation": operation, "error_type": type(exc).__name__},
            )
        )

    def put_bytes(
        self,
        data: bytes,
        *,
        expected_hash: str | None = None,
        media_type: str | None = None,
    ) -> ArtifactRecord:
        digest = hashlib.sha256(data).hexdigest()
        expected = _normalized_expected_hash(expected_hash)
        if expected is not None and expected != digest:
            exc = ArtifactIntegrityError(f"expected sha256 {expected}, computed {digest}")
            self._failure("put_bytes", exc)
            raise exc
        path = self._path_for_digest(digest)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                existing_digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if existing_digest != digest:
                    raise ArtifactIntegrityError(
                        f"immutable artifact path contains unexpected content: {path}"
                    )
            else:
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                    temporary_path = Path(handle.name)
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    os.replace(temporary_path, path)
                finally:
                    temporary_path.unlink(missing_ok=True)
        except (OSError, ArtifactIntegrityError) as exc:
            self._failure("put_bytes", exc)
            raise
        record = ArtifactRecord(
            artifact_ref=f"sha256:{digest}",
            content_hash=digest,
            hash_algorithm="sha256",
            byte_size=len(data),
            storage_path=str(path),
            created_at=datetime.now(timezone.utc),
            media_type=media_type,
        )
        self.event_sink.emit(
            StructuredEvent.create(
                "storage.artifact_put",
                "filesystem_artifact_store",
                entity_id=record.artifact_ref,
                attributes={"byte_size": record.byte_size, "hash_algorithm": "sha256"},
            )
        )
        return record

    def put_file(
        self,
        source: Path,
        *,
        expected_hash: str | None = None,
        media_type: str | None = None,
    ) -> ArtifactRecord:
        try:
            data = source.read_bytes()
        except OSError as exc:
            self._failure("put_file", exc)
            raise
        return self.put_bytes(data, expected_hash=expected_hash, media_type=media_type)

    def open(self, artifact_ref: str) -> BinaryIO:
        digest = self._digest_from_ref(artifact_ref)
        return self._path_for_digest(digest).open("rb")

    def get_bytes(self, artifact_ref: str) -> bytes:
        with self.open(artifact_ref) as handle:
            return handle.read()

    def verify(self, artifact_ref: str) -> bool:
        digest = self._digest_from_ref(artifact_ref)
        path = self._path_for_digest(digest)
        if not path.is_file():
            return False
        computed = hashlib.sha256(path.read_bytes()).hexdigest()
        verified = computed == digest
        self.event_sink.emit(
            StructuredEvent.create(
                "storage.artifact_verified" if verified else "storage.failure",
                "filesystem_artifact_store",
                entity_id=artifact_ref,
                severity="info" if verified else "error",
                attributes={"operation": "verify", "verified": verified},
            )
        )
        return verified


@dataclass(slots=True)
class JsonLineStateRepository:
    """Atomic single-record appends; no multi-record transaction guarantee."""

    root: Path
    event_sink: EventSink = field(default_factory=NullEventSink)
    _locks: dict[str, threading.Lock] = field(default_factory=dict, init=False, repr=False)
    _locks_guard: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _path(self, collection: str) -> Path:
        if not _COLLECTION.fullmatch(collection):
            raise ValueError(f"invalid state collection: {collection!r}")
        return self.root / f"{collection}.jsonl"

    def _lock_for(self, collection: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(collection, threading.Lock())

    def append(self, collection: str, record: dict[str, object]) -> None:
        path = self._path(collection)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        try:
            with self._lock_for(collection), path.open("a", encoding="utf-8") as handle:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    handle.write(line)
                    handle.flush()
                    os.fsync(handle.fileno())
                finally:
                    if fcntl is not None:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (OSError, TypeError, ValueError) as exc:
            self.event_sink.emit(
                StructuredEvent.create(
                    "storage.failure",
                    "jsonl_state_repository",
                    severity="error",
                    attributes={
                        "operation": "append",
                        "collection": collection,
                        "error_type": type(exc).__name__,
                    },
                )
            )
            raise

    def all(self, collection: str) -> list[dict[str, object]]:
        path = self._path(collection)
        if not path.exists():
            return []
        rows: list[dict[str, object]] = []
        with self._lock_for(collection), path.open("r", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                for line in handle:
                    if line.strip():
                        payload = json.loads(line)
                        if not isinstance(payload, dict):
                            raise ValueError(f"state record is not an object in {path}")
                        rows.append(payload)
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return rows

    def get_latest(
        self, collection: str, key: str, value: object
    ) -> dict[str, object] | None:
        for row in reversed(self.all(collection)):
            if row.get(key) == value:
                return row
        return None
