"""Restart transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RestartTransition:
    lineage_id: str
    reset_state: dict[str, Any]


def create_restart_state(
    prior_state: dict[str, Any] | None,
    lineage_id: str,
    stage_name: str,
    updated_at: str,
    reason: str,
) -> RestartTransition:
    prior = prior_state or {}
    reset_state = {
        "lineage_id": lineage_id,
        "parent_lineage_id": prior.get("parent_lineage_id"),
        "stage_name": stage_name,
        "status": "restarted",
        "trust_level": "low",
        "loop_index": 0,
        "latest_run_id": None,
        "best_checkpoint_ref": None,
        "last_stable_checkpoint_ref": None,
        "recent_failures": [],
        "known_pathologies": [reason],
        "last_decision": "restart_lineage",
        "last_decision_id": None,
        "branch_origin_checkpoint_ref": None,
        "child_lineage_ids": list(prior.get("child_lineage_ids", [])),
        "run_count": 0,
        "updated_at": updated_at,
    }
    return RestartTransition(lineage_id=lineage_id, reset_state=reset_state)
