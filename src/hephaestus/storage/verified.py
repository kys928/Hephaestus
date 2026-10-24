"""Read immutable JSON bytes once, verifying the same bytes that are parsed."""
import hashlib
import json
import re
from .base import ArtifactStore


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def read_verified_json(store: ArtifactStore, ref: str, *, limit: int = 16 * 1024 * 1024) -> dict:
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", ref):
        raise ValueError("evidence requires an immutable SHA-256 reference")
    with store.open(ref) as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit or "sha256:" + hashlib.sha256(raw).hexdigest() != ref:
        raise ValueError("immutable evidence bytes/hash mismatch")
    value = json.loads(raw, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError("evidence artifact must be a JSON object")
    return value
