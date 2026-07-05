"""Deterministic lightweight registry helpers for data artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_ref(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def as_dict(value: object | None) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    return {}


def as_str_list(value: object | None) -> list[str]:
    if not isinstance(value, list | tuple | set):
        return []
    return [str(item) for item in value]


@dataclass(slots=True)
class DataArtifactRegistry:
    """In-memory registry that stores refs, not artifact payloads."""

    run_id: str
    entries: list[dict[str, object]] = field(default_factory=list)

    def register(self, *, kind: str, ref: object | None, metadata: dict[str, object] | None = None) -> dict[str, object] | None:
        artifact_ref = as_ref(ref)
        if not artifact_ref:
            return None
        entry: dict[str, object] = {
            "run_id": self.run_id,
            "kind": str(kind),
            "ref": artifact_ref,
            "registered_at": utc_now_iso(),
            "metadata": dict(metadata or {}),
        }
        self.entries.append(entry)
        return entry

    def to_metadata(self) -> dict[str, object]:
        return {"artifact_refs": [dict(entry) for entry in self.entries]}
