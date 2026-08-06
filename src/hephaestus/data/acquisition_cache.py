"""Content-addressed cache and safe partial state for dataset acquisition."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from hephaestus.providers.datasets.acquisition import ProviderDatasetFile
from hephaestus.utils.hashing import hash_file, hash_json


def validate_relative_file_path(value: str) -> str:
    """Return a canonical provider path or reject traversal/absolute paths."""

    if not value or "\\" in value:
        raise ValueError("provider file path must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe provider file path: {value!r}")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class CacheLookup:
    status: str
    cache_key: str
    path: Path | None = None
    content_hash: str | None = None
    byte_size: int | None = None
    warning: str | None = None


@dataclass(frozen=True, slots=True)
class PartialDownloadState:
    cache_key: str
    path: Path
    metadata_path: Path
    provider_id: str
    dataset_id: str
    resolved_revision: str
    relative_path: str
    source_url: str
    etag: str | None
    byte_count: int
    provider_hash: str | None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        payload["metadata_path"] = str(self.metadata_path)
        return payload


@dataclass(slots=True)
class DatasetAcquisitionCache:
    root: Path

    def cache_key(
        self,
        *,
        provider_id: str,
        dataset_id: str,
        resolved_revision: str,
        file: ProviderDatasetFile,
    ) -> str:
        relative_path = validate_relative_file_path(file.relative_path)
        return hash_json(
            {
                "provider_id": provider_id,
                "dataset_id": dataset_id,
                "resolved_revision": resolved_revision,
                "relative_path": relative_path,
                "object_id": file.object_id,
                "provider_hash": file.provider_hash,
                "provider_hash_algorithm": file.provider_hash_algorithm,
            }
        )

    def _entry_path(self, cache_key: str) -> Path:
        return self.root / "entries" / cache_key[:2] / f"{cache_key}.json"

    def _object_path(self, digest: str) -> Path:
        return self.root / "objects" / "sha256" / digest[:2] / digest

    def _partial_paths(self, cache_key: str) -> tuple[Path, Path]:
        base = self.root / "partial" / cache_key[:2] / cache_key
        return base.with_suffix(".part"), base.with_suffix(".json")

    def lookup(
        self,
        *,
        provider_id: str,
        dataset_id: str,
        resolved_revision: str,
        file: ProviderDatasetFile,
    ) -> CacheLookup:
        key = self.cache_key(
            provider_id=provider_id,
            dataset_id=dataset_id,
            resolved_revision=resolved_revision,
            file=file,
        )
        entry_path = self._entry_path(key)
        if not entry_path.is_file():
            return CacheLookup("miss", key)
        try:
            payload = json.loads(entry_path.read_text(encoding="utf-8"))
            digest = str(payload["local_content_hash"]).removeprefix("sha256:")
            object_path = self._object_path(digest)
            if not object_path.is_file():
                return CacheLookup(
                    "corrupt", key, warning="cache entry object is missing"
                )
            computed = hash_file(object_path)
            expected_size = int(payload["byte_size"])
            if computed != digest or object_path.stat().st_size != expected_size:
                return CacheLookup(
                    "corrupt", key, warning="cache object failed content verification"
                )
            return CacheLookup(
                "hit",
                key,
                path=object_path,
                content_hash=f"sha256:{computed}",
                byte_size=expected_size,
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return CacheLookup(
                "corrupt", key, warning="cache entry metadata is malformed"
            )

    def load_partial(
        self,
        *,
        provider_id: str,
        dataset_id: str,
        resolved_revision: str,
        file: ProviderDatasetFile,
    ) -> PartialDownloadState | None:
        key = self.cache_key(
            provider_id=provider_id,
            dataset_id=dataset_id,
            resolved_revision=resolved_revision,
            file=file,
        )
        partial_path, metadata_path = self._partial_paths(key)
        if not partial_path.is_file() or not metadata_path.is_file():
            return None
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            state = PartialDownloadState(
                cache_key=key,
                path=partial_path,
                metadata_path=metadata_path,
                provider_id=str(payload["provider_id"]),
                dataset_id=str(payload["dataset_id"]),
                resolved_revision=str(payload["resolved_revision"]),
                relative_path=str(payload["relative_path"]),
                source_url=str(payload["source_url"]),
                etag=payload.get("etag"),
                byte_count=int(payload["byte_count"]),
                provider_hash=payload.get("provider_hash"),
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            self.clear_partial_by_key(key)
            return None
        expected = {
            "provider_id": provider_id,
            "dataset_id": dataset_id,
            "resolved_revision": resolved_revision,
            "relative_path": validate_relative_file_path(file.relative_path),
            "source_url": file.source_url,
            "provider_hash": file.provider_hash,
        }
        actual = {key_name: getattr(state, key_name) for key_name in expected}
        if actual != expected or state.byte_count != partial_path.stat().st_size:
            self.clear_partial_by_key(key)
            return None
        return state

    def prepare_partial(
        self,
        *,
        provider_id: str,
        dataset_id: str,
        resolved_revision: str,
        file: ProviderDatasetFile,
        byte_count: int,
        etag: str | None,
    ) -> PartialDownloadState:
        key = self.cache_key(
            provider_id=provider_id,
            dataset_id=dataset_id,
            resolved_revision=resolved_revision,
            file=file,
        )
        partial_path, metadata_path = self._partial_paths(key)
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.touch(exist_ok=True)
        state = PartialDownloadState(
            cache_key=key,
            path=partial_path,
            metadata_path=metadata_path,
            provider_id=provider_id,
            dataset_id=dataset_id,
            resolved_revision=resolved_revision,
            relative_path=validate_relative_file_path(file.relative_path),
            source_url=file.source_url,
            etag=etag,
            byte_count=byte_count,
            provider_hash=file.provider_hash,
        )
        self._write_json_atomic(metadata_path, state.to_dict())
        return state

    def clear_partial_by_key(self, cache_key: str) -> dict[str, object]:
        partial_path, metadata_path = self._partial_paths(cache_key)
        removed: list[str] = []
        for path in (partial_path, metadata_path):
            if path.exists():
                path.unlink(missing_ok=True)
                removed.append(str(path))
        return {"cache_key": cache_key, "action": "partial_removed", "removed": removed}

    def quarantine_corrupt(self, cache_key: str) -> dict[str, object]:
        """Move proven-corrupt cache metadata/content aside without deleting evidence."""

        entry_path = self._entry_path(cache_key)
        moved: list[str] = []
        quarantine = self.root / "quarantine" / cache_key[:2]
        quarantine.mkdir(parents=True, exist_ok=True)
        if entry_path.is_file():
            try:
                payload = json.loads(entry_path.read_text(encoding="utf-8"))
                digest = str(payload.get("local_content_hash", "")).removeprefix(
                    "sha256:"
                )
                object_path = self._object_path(digest) if digest else None
                if object_path is not None and object_path.is_file():
                    destination = quarantine / f"{cache_key}-{digest}.corrupt"
                    if not destination.exists():
                        os.replace(object_path, destination)
                        moved.append(str(destination))
                entry_destination = quarantine / f"{cache_key}.entry.corrupt"
                os.replace(entry_path, entry_destination)
                moved.append(str(entry_destination))
            except (OSError, ValueError, json.JSONDecodeError):
                entry_destination = quarantine / f"{cache_key}.entry.corrupt"
                if entry_path.exists() and not entry_destination.exists():
                    os.replace(entry_path, entry_destination)
                    moved.append(str(entry_destination))
        return {
            "cache_key": cache_key,
            "action": "corrupt_cache_quarantined",
            "moved": moved,
        }

    def store_completed(
        self,
        *,
        provider_id: str,
        dataset_id: str,
        resolved_revision: str,
        file: ProviderDatasetFile,
        source: Path,
        local_content_hash: str,
        byte_size: int,
    ) -> CacheLookup:
        digest = local_content_hash.removeprefix("sha256:")
        if hash_file(source) != digest or source.stat().st_size != byte_size:
            raise ValueError(
                "completed cache source failed local integrity verification"
            )
        key = self.cache_key(
            provider_id=provider_id,
            dataset_id=dataset_id,
            resolved_revision=resolved_revision,
            file=file,
        )
        object_path = self._object_path(digest)
        object_path.parent.mkdir(parents=True, exist_ok=True)
        if object_path.exists():
            if hash_file(object_path) != digest:
                raise ValueError("immutable cache object contains corrupt content")
            source.unlink(missing_ok=True)
        else:
            os.replace(source, object_path)
        entry = {
            "cache_key": key,
            "provider_id": provider_id,
            "dataset_id": dataset_id,
            "resolved_revision": resolved_revision,
            "relative_path": validate_relative_file_path(file.relative_path),
            "provider_hash": file.provider_hash,
            "provider_hash_algorithm": file.provider_hash_algorithm,
            "local_content_hash": f"sha256:{digest}",
            "byte_size": byte_size,
            "object_ref": str(object_path),
            "status": "complete",
        }
        self._write_json_atomic(self._entry_path(key), entry)
        _, metadata_path = self._partial_paths(key)
        metadata_path.unlink(missing_ok=True)
        return CacheLookup("stored", key, object_path, f"sha256:{digest}", byte_size)

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


__all__ = [
    "CacheLookup",
    "DatasetAcquisitionCache",
    "PartialDownloadState",
    "validate_relative_file_path",
]
