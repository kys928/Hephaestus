"""Small deterministic JSON/text IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hephaestus.utils.paths import ensure_parent


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, content: str) -> Path:
    target = ensure_parent(path)
    target.write_text(content, encoding="utf-8")
    return target


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("json_document_must_be_object")
    return payload


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return target


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = ensure_parent(path)
    tmp = target.with_name(f".{target.name}.tmp")
    write_json(tmp, payload)
    tmp.replace(target)
    return target
