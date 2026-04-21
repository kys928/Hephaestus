from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hephaestus.state._json_store import JsonStore


@dataclass(slots=True)
class DecisionStore:
    root: Path

    def append(self, record: dict[str, Any]) -> None:
        JsonStore(self.root, "decision_records.jsonl").append(record)

    def append_approval_request(self, record: dict[str, Any]) -> None:
        JsonStore(self.root, "approval_requests.jsonl").append(record)

    def append_approval_decision(self, record: dict[str, Any]) -> None:
        JsonStore(self.root, "approval_decisions.jsonl").append(record)

    def all(self) -> list[dict[str, Any]]:
        return JsonStore(self.root, "decision_records.jsonl").all()

    def all_approval_requests(self) -> list[dict[str, Any]]:
        return JsonStore(self.root, "approval_requests.jsonl").all()

    def all_approval_decisions(self) -> list[dict[str, Any]]:
        return JsonStore(self.root, "approval_decisions.jsonl").all()

    def get(self, decision_id: str) -> dict[str, Any] | None:
        return JsonStore(self.root, "decision_records.jsonl").get_latest("decision_id", decision_id)

    def get_approval_request(self, request_id: str) -> dict[str, Any] | None:
        return JsonStore(self.root, "approval_requests.jsonl").get_latest("request_id", request_id)

    def get_latest_approval_decision(self, request_id: str) -> dict[str, Any] | None:
        return JsonStore(self.root, "approval_decisions.jsonl").get_latest("request_id", request_id)
