from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hephaestus.state._json_store import JsonStore


@dataclass(slots=True)
class RunStore:
    root: Path

    def append(self, record: dict[str, Any]) -> None:
        JsonStore(self.root, "run_records.jsonl").append(record)

    def all(self) -> list[dict[str, Any]]:
        return JsonStore(self.root, "run_records.jsonl").all()

    def get(self, run_id: str) -> dict[str, Any] | None:
        return JsonStore(self.root, "run_records.jsonl").get_latest("run_id", run_id)
