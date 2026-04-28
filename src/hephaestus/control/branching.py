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
    parent_trust = str(parent.get("trust_level", "unknown"))
    child_trust = "medium" if parent_trust in {"medium", "high", "certified"} else "low"
    origin_run_id = parent.get("origin_run_id") or parent.get("latest_run_id")
    child_state = {
        "lineage_id": child_lineage_id,
        "parent_lineage_id": parent.get("lineage_id"),
        "child_lineage_ids": [],
        "stage_name": stage_name,
        "status": "exploratory",
        "trust_level": child_trust,
        "origin_run_id": origin_run_id,
        "origin_checkpoint_ref": origin_checkpoint_ref,
        "branch_origin_checkpoint_ref": origin_checkpoint_ref,
        "created_at": updated_at,
        "updated_at": updated_at,
        "architecture_contract_ref": parent.get("architecture_contract_ref"),
        "tokenizer_contract_ref": parent.get("tokenizer_contract_ref"),
        "data_policy_ref": parent.get("data_policy_ref"),
        "training_recipe_ref": parent.get("training_recipe_ref"),
        "eval_policy_ref": parent.get("eval_policy_ref"),
        "loop_index": 0,
        "run_count": 0,
        "latest_run_id": None,
        "best_checkpoint_ref": origin_checkpoint_ref,
        "last_stable_checkpoint_ref": None,
        "certified_stable_checkpoint_ref": None,
        "last_certification_result": "certification_not_eligible",
        "last_decision": "branch_new_experiment",
        "last_decision_id": None,
        "last_requested_action": "branch_new_experiment",
        "last_effective_action": "branch_new_experiment",
        "last_approval_status": "approved",
        "pending_approval": False,
        "last_high_impact_request_id": None,
        "last_repeated_eval_count": 0,
        "last_consistency_score": 0.0,
        "last_variance_risk": "unknown",
        "certification_recheck_count": 0,
        "repeatability_sufficient": False,
        "recent_failures": [],
        "known_pathologies": [],
        "major_interventions": [],
        "metadata": {"lineage_event": "branch_created"},
    }
    return BranchTransition(child_lineage_id=child_lineage_id, child_state=child_state)
