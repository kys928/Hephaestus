from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hephaestus.state._json_store import JsonSingleDocument


@dataclass(slots=True)
class LineageStore:
    root: Path

    def _all_doc(self) -> JsonSingleDocument:
        return JsonSingleDocument(self.root, "lineage_states.json")

    def _legacy_doc(self) -> JsonSingleDocument:
        return JsonSingleDocument(self.root, "lineage_state.json")

    def _read_all(self) -> dict[str, dict[str, Any]]:
        return self._all_doc().read() or {}

    def _write_all(self, states: dict[str, dict[str, Any]]) -> None:
        self._all_doc().write(states)

    def set_current(self, record: dict[str, Any]) -> None:
        lineage_id = str(record["lineage_id"])
        states = self._read_all()
        states[lineage_id] = record
        self._write_all(states)
        self._legacy_doc().write(record)

    def get_current(self, lineage_id: str | None = None) -> dict[str, Any] | None:
        if lineage_id is None:
            return self._legacy_doc().read()
        return self._read_all().get(lineage_id)

    def all(self) -> dict[str, dict[str, Any]]:
        return self._read_all()

    def add_child(self, parent_lineage_id: str, child_lineage_id: str) -> None:
        states = self._read_all()
        parent = states.get(parent_lineage_id)
        if parent is None:
            return
        children = [str(item) for item in parent.get("child_lineage_ids", [])]
        if child_lineage_id not in children:
            children.append(child_lineage_id)
            parent["child_lineage_ids"] = children
            states[parent_lineage_id] = parent
            self._write_all(states)
            legacy = self._legacy_doc().read()
            if legacy and legacy.get("lineage_id") == parent_lineage_id:
                self._legacy_doc().write(parent)
