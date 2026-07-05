"""Deterministic hashing helpers for JSON payloads and artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize a JSON-compatible payload to stable UTF-8 bytes."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest for bytes."""

    return hashlib.sha256(data).hexdigest()


def hash_json(payload: Any) -> str:
    """Return a deterministic SHA-256 hash of a JSON-serializable payload."""

    return sha256_bytes(canonical_json_bytes(payload))


def hash_text(text: str) -> str:
    """Return the SHA-256 hex digest of UTF-8 text."""

    return sha256_bytes(text.encode("utf-8"))


def hash_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 hex digest for a file without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash_record(path: str | Path, hash_type: str = "sha256") -> dict[str, str]:
    """Return a JSON-serializable content-hash record for an artifact path."""

    if hash_type != "sha256":
        raise ValueError(f"unsupported_hash_type={hash_type}")
    artifact = Path(path)
    return {"artifact_ref": str(artifact), "hash_type": hash_type, "content_hash": hash_file(artifact)}
