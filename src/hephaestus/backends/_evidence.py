from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from hephaestus.schemas.runtime_event import RuntimeEvent, RuntimeEventCategory


def file_sha256(ref: str) -> str:
    digest = hashlib.sha256()
    with Path(ref).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(ref: str) -> dict[str, Any]:
    payload = json.loads(Path(ref).read_text())
    return payload if isinstance(payload, dict) else {}


def extract_metric_value(metrics: dict[str, Any], name: str, default: float) -> float:
    try:
        return float(metrics.get(name, default))
    except (TypeError, ValueError):
        return default


def normalize_checkpoint_candidate(ref: str, *, probe_score: float | None = None, step: int | None = None) -> dict[str, Any]:
    candidate: dict[str, Any] = {"checkpoint_ref": ref, "step": step, "probe_score": probe_score, "score": probe_score}
    path = Path(ref)
    if path.exists():
        candidate.update({"content_hash": file_sha256(ref), "hash_type": "sha256", "integrity_level": "content_hash"})
    else:
        candidate.update({"content_hash": None, "hash_type": None, "integrity_level": "ref"})
    return candidate


def incident(run_id: str, suffix: str, message: str, payload_ref: str | None = None) -> RuntimeEvent:
    return RuntimeEvent(event_id=f"{run_id}-{suffix}", run_id=run_id, step=0, category=RuntimeEventCategory.INCIDENT, message=message, payload_ref=payload_ref)


def status_event(run_id: str, suffix: str, message: str, payload_ref: str | None = None) -> RuntimeEvent:
    return RuntimeEvent(event_id=f"{run_id}-{suffix}", run_id=run_id, step=0, category=RuntimeEventCategory.STATUS, message=message, payload_ref=payload_ref)
