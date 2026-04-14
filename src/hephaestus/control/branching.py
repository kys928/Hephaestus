"""Branching transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class BranchTransition:
    child_lineage_id: str
    child_state: dict[str, Any]


def create_branch_state(
    parent_state: dict[str, Any] | None,
    child_lineage_id: str,
    stage_name: str,
    origin_checkpoint_ref: str | None,
    updated_at: str,
) -> BranchTransition:
    parent = parent_state or {}
    child_state = {
        "lineage_id": child_lineage_id,
        "parent_lineage_id": parent.get("lineage_id"),
        "stage_name": stage_name,
        "status": "active",
        "trust_level": "medium",
        "loop_index": 0,
        "latest_run_id": None,
        "best_checkpoint_ref": origin_checkpoint_ref,
        "last_stable_checkpoint_ref": parent.get("last_stable_checkpoint_ref"),
        "recent_failures": [],
        "known_pathologies": [],
        "last_decision": "branch_new_experiment",
        "last_decision_id": None,
        "branch_origin_checkpoint_ref": origin_checkpoint_ref,
        "child_lineage_ids": [],
        "run_count": 0,
        "updated_at": updated_at,
    }
    return BranchTransition(child_lineage_id=child_lineage_id, child_state=child_state)
