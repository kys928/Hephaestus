from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hephaestus.state._json_store import JsonSingleDocument


@dataclass(slots=True)
class LineageStore:
    root: Path

    def set_current(self, record: dict[str, Any]) -> None:
        JsonSingleDocument(self.root, "lineage_state.json").write(record)

    def get_current(self) -> dict[str, Any] | None:
        return JsonSingleDocument(self.root, "lineage_state.json").read()
