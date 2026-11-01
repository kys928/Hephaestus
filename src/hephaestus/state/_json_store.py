from __future__ import annotations

import json
import os  # Retain the existing atomic-replace fault-injection surface.
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from hephaestus.storage.durability import atomic_write_bytes, append_bytes
from .campaign_witness import state_directory_owner


@dataclass(slots=True)
class JsonStore:
    root: Path
    filename: str

    def _path(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root / self.filename

    def append(self, record: dict[str, Any]) -> None:
        with state_directory_owner(self.root):
            path = self._path()
            append_bytes(path, (json.dumps(record, sort_keys=True) + "\n").encode("utf-8"))

    def all(self) -> list[dict[str, Any]]:
        path = self._path()
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def get_latest(self, key: str, value: str) -> dict[str, Any] | None:
        for row in reversed(self.all()):
            if row.get(key) == value:
                return row
        return None


@dataclass(slots=True)
class JsonSingleDocument:
    root: Path
    filename: str

    def _path(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root / self.filename

    def write(self, record: dict[str, Any]) -> None:
        # A controller crash must not leave the authoritative lineage JSON
        # truncated. Write and sync a sibling file, then atomically replace it.
        with state_directory_owner(self.root):
            path = self._path()
            atomic_write_bytes(path, json.dumps(record, indent=2, sort_keys=True).encode("utf-8"))

    def read(self) -> dict[str, Any] | None:
        path = self._path()
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
