from __future__ import annotations

from dataclasses import dataclass

from hephaestus.schemas.decision_record import DecisionRecord
from hephaestus.schemas.judge_entry import JudgeEntry


@dataclass(slots=True)
class JudgeEntryRole:
    name: str = "judge_entry"

    def run(self, run_id: str, lineage_id: str, stage_name: str, created_at: str) -> tuple[JudgeEntry, DecisionRecord]:
        entry = JudgeEntry(
            run_id=run_id,
            lineage_id=lineage_id,
            objective="Run one bounded training/eval cycle with safety gates.",
            constraints=[
                "preserve_role_boundaries",
                "compact_state_only",
                "deterministic_regressions_block_promotion",
            ],
            approved=True,
            stage_name=stage_name,
        )
        decision = DecisionRecord(
            decision_id=f"dec-{run_id}-entry",
            run_id=run_id,
            lineage_id=lineage_id,
            role="judge_entry",
            action="approve_run",
            rationale="stage constraints satisfied",
            confidence=0.8,
            created_at=created_at,
        )
        return entry, decision
