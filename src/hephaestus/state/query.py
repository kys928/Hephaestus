"""Filesystem-backed lineage query helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hephaestus.state.decision_store import DecisionStore
from hephaestus.state.lineage_store import LineageStore
from hephaestus.state.run_store import RunStore


@dataclass(slots=True)
class Query:
    root: Path

    def _runs(self) -> RunStore:
        return RunStore(self.root)

    def _decisions(self) -> DecisionStore:
        return DecisionStore(self.root)

    def _lineages(self) -> LineageStore:
        return LineageStore(self.root)

    def latest_run_in_lineage(self, lineage_id: str) -> dict[str, Any] | None:
        rows = [row for row in self._runs().all() if row.get("lineage_id") == lineage_id]
        return rows[-1] if rows else None

    def recent_failures(self, lineage_id: str, limit: int = 3) -> list[dict[str, Any]]:
        rows = [row for row in self._runs().all() if row.get("lineage_id") == lineage_id and row.get("status") != "completed"]
        return rows[-limit:]

    def runs_in_stage(self, lineage_id: str, stage_name: str) -> list[dict[str, Any]]:
        return [
            row
            for row in self._runs().all()
            if row.get("lineage_id") == lineage_id and row.get("stage_name") == stage_name
        ]

    def recent_decisions(self, lineage_id: str, limit: int = 5) -> list[dict[str, Any]]:
        rows = [row for row in self._decisions().all() if row.get("lineage_id") == lineage_id]
        return rows[-limit:]

    def best_checkpoint(self, lineage_id: str) -> str | None:
        lineage = self._lineages().get_current(lineage_id)
        return None if not lineage else lineage.get("best_checkpoint_ref")

    def last_stable_checkpoint(self, lineage_id: str) -> str | None:
        lineage = self._lineages().get_current(lineage_id)
        return None if not lineage else lineage.get("last_stable_checkpoint_ref")

    def lineage_relationships(self, lineage_id: str) -> dict[str, Any]:
        lineage = self._lineages().get_current(lineage_id) or {}
        return {
            "lineage_id": lineage_id,
            "parent_lineage_id": lineage.get("parent_lineage_id"),
            "child_lineage_ids": list(lineage.get("child_lineage_ids", [])),
        }
