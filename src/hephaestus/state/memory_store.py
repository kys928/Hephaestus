from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hephaestus.schemas.memory_record import MemoryRecord
from hephaestus.state._json_store import JsonStore


@dataclass(slots=True)
class MemoryStore:
    root: Path

    def append(self, record: dict[str, object] | MemoryRecord) -> None:
        normalized = record if isinstance(record, MemoryRecord) else MemoryRecord.from_dict(dict(record))
        if self.get(normalized.memory_id):
            return
        JsonStore(self.root, "memory_records.jsonl").append(normalized.to_dict())

    def get(self, memory_id: str) -> dict[str, object] | None:
        row = JsonStore(self.root, "memory_records.jsonl").get_latest("memory_id", memory_id)
        return MemoryRecord.from_dict(row).to_dict() if row else None

    def list_all(self) -> list[dict[str, object]]:
        return [MemoryRecord.from_dict(row).to_dict() for row in JsonStore(self.root, "memory_records.jsonl").all()]

    def list_for_run(self, run_id: str) -> list[dict[str, object]]:
        return [row for row in self.list_all() if str(row.get("run_id") or "") == run_id]

    def list_for_lineage(self, lineage_id: str) -> list[dict[str, object]]:
        return [row for row in self.list_all() if str(row.get("lineage_id") or "") == lineage_id]

    def find_by_type(self, memory_type: str) -> list[dict[str, object]]:
        return [row for row in self.list_all() if str(row.get("memory_type") or "") == memory_type]

    def find_by_tag(self, tag: str) -> list[dict[str, object]]:
        return [row for row in self.list_all() if tag in [str(item) for item in row.get("tags", [])]]

    def recent(self, limit: int = 20) -> list[dict[str, object]]:
        rows = self.list_all()
        return rows[-max(limit, 0) :]
