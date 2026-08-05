from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from hephaestus.storage import (
    FileSystemArtifactStore,
    InMemoryLockProvider,
    JsonLineStateRepository,
)
from hephaestus.storage.filesystem import ArtifactIntegrityError


def test_artifact_store_uses_immutable_hash_identity_and_verifies_content(tmp_path) -> None:
    store = FileSystemArtifactStore(tmp_path / "artifacts")
    data = b"bounded artifact"
    digest = hashlib.sha256(data).hexdigest()

    first = store.put_bytes(data, expected_hash=digest, media_type="text/plain")
    duplicate = store.put_bytes(data)

    assert first.artifact_ref == f"sha256:{digest}"
    assert duplicate.artifact_ref == first.artifact_ref
    assert store.get_bytes(first.artifact_ref) == data
    assert store.verify(first.artifact_ref) is True
    assert list((tmp_path / "artifacts" / "objects" / "sha256").rglob(digest))


def test_artifact_store_rejects_hash_mismatch_and_detects_corruption(tmp_path) -> None:
    store = FileSystemArtifactStore(tmp_path)
    with pytest.raises(ArtifactIntegrityError):
        store.put_bytes(b"data", expected_hash="0" * 64)

    record = store.put_bytes(b"original")
    path = tmp_path / "objects" / "sha256" / record.content_hash[:2] / record.content_hash
    path.write_bytes(b"corrupted")
    assert store.verify(record.artifact_ref) is False


def test_jsonl_repository_supports_concurrent_single_record_appends(tmp_path) -> None:
    repository = JsonLineStateRepository(tmp_path)

    def append(index: int) -> None:
        repository.append("jobs", {"job_id": f"job-{index}", "index": index})

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(append, range(100)))

    rows = repository.all("jobs")
    assert len(rows) == 100
    assert {row["index"] for row in rows} == set(range(100))
    assert repository.get_latest("jobs", "job_id", "job-42") == {
        "index": 42,
        "job_id": "job-42",
    }


def test_local_lock_enforces_owner_and_expiration_without_distributed_claims() -> None:
    locks = InMemoryLockProvider()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first = locks.acquire("artifact:index", "worker-a", 10, now=now)

    assert first is not None
    assert locks.acquire("artifact:index", "worker-b", 10, now=now) is None
    assert locks.release(first) is True
    second = locks.acquire("artifact:index", "worker-b", 10, now=now + timedelta(seconds=1))
    assert second is not None
