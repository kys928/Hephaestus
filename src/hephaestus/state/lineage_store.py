from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hephaestus.schemas.lineage_state import LineageState
from hephaestus.state._json_store import JsonSingleDocument


@dataclass(slots=True)
class LineageStore:
    root: Path

    def _all_doc(self) -> JsonSingleDocument:
        return JsonSingleDocument(self.root, "lineage_states.json")

    def _legacy_doc(self) -> JsonSingleDocument:
        return JsonSingleDocument(self.root, "lineage_state.json")

    def _read_all(self) -> dict[str, dict[str, Any]]:
        raw = self._all_doc().read() or {}
        normalized: dict[str, dict[str, Any]] = {}
        for lineage_id, payload in raw.items():
            state = LineageState.from_dict(dict(payload))
            if not state.lineage_id:
                state.lineage_id = str(lineage_id)
            normalized[state.lineage_id] = state.to_dict()
        return normalized

    def _write_all(self, states: dict[str, dict[str, Any]]) -> None:
        ordered = {lineage_id: states[lineage_id] for lineage_id in sorted(states)}
        self._all_doc().write(ordered)

    def set_current(self, record: dict[str, Any]) -> None:
        state = LineageState.from_dict(dict(record))
        lineage_id = str(state.lineage_id)
        states = self._read_all()
        states[lineage_id] = state.to_dict()
        self._write_all(states)
        self._legacy_doc().write(state.to_dict())

    def get_current(self, lineage_id: str | None = None) -> dict[str, Any] | None:
        if lineage_id is None:
            legacy = self._legacy_doc().read()
            if not legacy:
                return None
            return LineageState.from_dict(legacy).to_dict()
        state = self._read_all().get(lineage_id)
        return None if state is None else LineageState.from_dict(state).to_dict()

    def list_lineages(self) -> dict[str, dict[str, Any]]:
        return self._read_all()

    def all(self) -> dict[str, dict[str, Any]]:
        return self.list_lineages()

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

    def get_children(self, lineage_id: str) -> list[str]:
        state = self.get_current(lineage_id)
        if not state:
            return []
        return [str(item) for item in state.get("child_lineage_ids", [])]

    def get_parent(self, lineage_id: str) -> str | None:
        state = self.get_current(lineage_id)
        if not state:
            return None
        parent = state.get("parent_lineage_id")
        return str(parent) if parent else None
