from __future__ import annotations

from dataclasses import dataclass

from hephaestus.policy.judge_policy import JudgePolicy
from hephaestus.schemas.decision_record import DecisionRecord
from hephaestus.schemas.judge_entry import JudgeEntry


@dataclass(slots=True)
class JudgeEntryRole:
    judge_policy: JudgePolicy
    name: str = "judge_entry"

    def run(
        self,
        run_id: str,
        lineage_id: str,
        stage_name: str,
        created_at: str,
        lineage_state: dict[str, object] | None,
        recent_failures: list[dict[str, object]],
    ) -> tuple[JudgeEntry, DecisionRecord]:
        state = lineage_state or {}
        mode = self.judge_policy.decide_entry_mode(
            lineage_status=str(state.get("status", "active")),
            recent_failure_count=len(recent_failures),
            best_checkpoint_ref=state.get("best_checkpoint_ref") if state else None,
            last_stable_checkpoint_ref=state.get("last_stable_checkpoint_ref") if state else None,
            parent_lineage_id=state.get("parent_lineage_id") if state else None,
        )
        entry = JudgeEntry(
            run_id=run_id,
            lineage_id=lineage_id,
            objective="Run one bounded training/eval cycle with safety gates.",
            constraints=[
                "preserve_role_boundaries",
                "compact_state_only",
                "deterministic_regressions_block_promotion",
                f"entry_mode={mode.value}",
            ],
            approved=True,
            stage_name=stage_name,
            entry_mode=mode,
        )
        decision = DecisionRecord(
            decision_id=f"dec-{run_id}-entry",
            run_id=run_id,
            lineage_id=lineage_id,
            role="judge_entry",
            action=mode.value,
            rationale="entry mode selected from lineage truth and recent outcomes",
            confidence=0.8,
            created_at=created_at,
            metadata={
                "recent_failure_count": len(recent_failures),
                "best_checkpoint_ref": state.get("best_checkpoint_ref") if state else None,
                "last_stable_checkpoint_ref": state.get("last_stable_checkpoint_ref") if state else None,
            },
        )
        return entry, decision
