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

    def certified_stable_checkpoint(self, lineage_id: str) -> str | None:
        lineage = self._lineages().get_current(lineage_id)
        return None if not lineage else lineage.get("certified_stable_checkpoint_ref")

    def last_certification_decision(self, lineage_id: str) -> str | None:
        lineage = self._lineages().get_current(lineage_id)
        return None if not lineage else lineage.get("last_certification_result")

    def recent_failed_certifications(self, lineage_id: str, limit: int = 5) -> list[dict[str, Any]]:
        rows = [
            row
            for row in self.recent_decisions(lineage_id, limit=100)
            if "certification_state=certification_blocked_by_regression" in str(row.get("rationale", ""))
        ]
        return rows[-limit:]

    def recent_inconclusive_promotions(self, lineage_id: str, limit: int = 5) -> list[dict[str, Any]]:
        states = {
            "certification_inconclusive",
            "certification_recheck_required",
            "certification_inconclusive_due_to_variance",
        }
        rows = [
            row
            for row in self.recent_decisions(lineage_id, limit=100)
            if any(f"certification_state={state}" in str(row.get("rationale", "")) for state in states)
        ]
        return rows[-limit:]

    def recent_certification_attempts_for_checkpoint(
        self,
        lineage_id: str,
        checkpoint_ref: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        rows = [
            row
            for row in self.recent_decisions(lineage_id, limit=200)
            if row.get("role") == "judge_exit" and str(row.get("metadata", {}).get("checkpoint_ref", "")) == checkpoint_ref
        ]
        return rows[-limit:]

    def checkpoint_repeatability_summary(self, lineage_id: str, checkpoint_ref: str | None = None) -> dict[str, Any]:
        lineage = self._lineages().get_current(lineage_id) or {}
        target = checkpoint_ref or str(lineage.get("best_checkpoint_ref", ""))
        if not target:
            return {
                "checkpoint_ref": None,
                "attempt_count": 0,
                "repeated_eval_count": 0,
                "recent_inconclusive": 0,
                "recent_inconsistency_signals": 0,
            }

        attempts = self.recent_certification_attempts_for_checkpoint(lineage_id, target, limit=20)
        inconclusive = 0
        inconsistency = 0
        repeated_count = 0
        for attempt in attempts:
            metadata = dict(attempt.get("metadata", {}))
            state = str(metadata.get("certification_state", ""))
            if state in {
                "certification_inconclusive",
                "certification_recheck_required",
                "certification_inconclusive_due_to_variance",
            }:
                inconclusive += 1
            if state == "certification_blocked_by_inconsistency":
                inconsistency += 1
            repeated_count = max(repeated_count, int(metadata.get("repeated_eval_count", 0)))

        return {
            "checkpoint_ref": target,
            "attempt_count": len(attempts),
            "repeated_eval_count": repeated_count,
            "recent_inconclusive": inconclusive,
            "recent_inconsistency_signals": inconsistency,
            "recent_variance_signals": sum(
                1 for attempt in attempts if str(attempt.get("metadata", {}).get("variance_risk", "")) == "high"
            ),
        }
