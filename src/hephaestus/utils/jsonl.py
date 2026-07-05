"""JSON Lines helpers for append-only state records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from hephaestus.utils.paths import ensure_parent


def encode_jsonl_record(record: dict[str, Any]) -> str:
    """Encode one JSONL record with deterministic key order."""

    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def append_jsonl(path: str | Path, record: dict[str, Any]) -> Path:
    target = ensure_parent(path)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(encode_jsonl_record(record))
    return target


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"jsonl_record_must_be_object line={line_number}")
            rows.append(payload)
    return rows


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(encode_jsonl_record(record))
    return target


def latest_by_key(path: str | Path, key: str, value: object) -> dict[str, Any] | None:
    for row in reversed(read_jsonl(path)):
        if row.get(key) == value:
            return row
    return None
