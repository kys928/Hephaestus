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
    interventions = [dict(item) for item in prior.get("major_interventions", []) if isinstance(item, dict)]
    interventions.append(
        {
            "type": "restart",
            "run_id": prior.get("latest_run_id"),
            "reason": reason,
            "created_at": updated_at,
        }
    )
    pathologies = [str(item) for item in prior.get("known_pathologies", [])]
    if reason not in pathologies:
        pathologies.append(reason)
    reset_state = {
        "lineage_id": lineage_id,
        "parent_lineage_id": prior.get("parent_lineage_id"),
        "child_lineage_ids": list(prior.get("child_lineage_ids", [])),
        "stage_name": stage_name,
        "status": "suspect",
        "trust_level": "low",
        "origin_run_id": prior.get("origin_run_id") or prior.get("latest_run_id"),
        "origin_checkpoint_ref": prior.get("origin_checkpoint_ref"),
        "branch_origin_checkpoint_ref": prior.get("branch_origin_checkpoint_ref"),
        "created_at": prior.get("created_at") or updated_at,
        "updated_at": updated_at,
        "architecture_contract_ref": prior.get("architecture_contract_ref"),
        "tokenizer_contract_ref": prior.get("tokenizer_contract_ref"),
        "data_policy_ref": prior.get("data_policy_ref"),
        "training_recipe_ref": prior.get("training_recipe_ref"),
        "eval_policy_ref": prior.get("eval_policy_ref"),
        "loop_index": 0,
        "run_count": 0,
        "latest_run_id": None,
        "best_checkpoint_ref": None,
        "last_stable_checkpoint_ref": None,
        "certified_stable_checkpoint_ref": None,
        "last_certification_result": "certification_not_eligible",
        "last_decision": "restart_lineage",
        "last_decision_id": None,
        "last_requested_action": "restart_lineage",
        "last_effective_action": "restart_lineage",
        "last_approval_status": "approved",
        "pending_approval": False,
        "last_high_impact_request_id": None,
        "last_repeated_eval_count": 0,
        "last_consistency_score": 0.0,
        "last_variance_risk": "unknown",
        "certification_recheck_count": 0,
        "repeatability_sufficient": False,
        "recent_failures": [],
        "known_pathologies": pathologies[-10:],
        "major_interventions": interventions[-20:],
        "metadata": {**dict(prior.get("metadata", {})), "lineage_event": "restart"},
    }
    return RestartTransition(lineage_id=lineage_id, reset_state=reset_state)
