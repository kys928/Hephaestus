from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class JsonStore:
    root: Path
    filename: str

    def _path(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root / self.filename

    def append(self, record: dict[str, Any]) -> None:
        path = self._path()
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

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
        path = self._path()
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def read(self) -> dict[str, Any] | None:
        path = self._path()
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
