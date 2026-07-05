from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hephaestus.state._json_store import JsonStore


@dataclass(slots=True)
class OperatorActionStore:
    root: Path

    def append(self, record: dict[str, Any]) -> None:
        JsonStore(self.root, "operator_actions.jsonl").append(record)

    def all(self) -> list[dict[str, Any]]:
        return JsonStore(self.root, "operator_actions.jsonl").all()

    def get(self, action_event_id: str) -> dict[str, Any] | None:
        return JsonStore(self.root, "operator_actions.jsonl").get_latest("action_event_id", action_event_id)

    def list_for_run(self, run_id: str) -> list[dict[str, Any]]:
        return [row for row in self.all() if row.get("run_id") == run_id]
