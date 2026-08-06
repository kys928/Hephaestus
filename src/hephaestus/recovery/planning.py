"""Bounded recovery recommendation planning without action execution."""

from __future__ import annotations

import hashlib

from hephaestus.recovery.models import (
    CheckpointRecoveryDecision,
    FailureClassification,
    NormalizedFailureEvidence,
    RecoveryRecommendation,
    RecoveryRequest,
    RetryBudgetDecision,
)

_REGISTRY_ACTION = {
    "retry_same_operation": "rerun_same_config",
    "retry_after_backoff": "rerun_same_config",
    "resume_verified_checkpoint": "continue_from_checkpoint",
    "restart_bounded_job": "rerun_same_config",
    "request_replacement_worker": "rerun_same_config",
    "reacquire_partial_artifact": "rerun_same_config",
    "rerun_evaluation": "request_recheck",
    "collect_more_evidence": "request_recheck",
    "rollback_verified_checkpoint": "rollback_to_checkpoint",
    "branch_new_experiment": "branch_new_experiment",
    "quarantine_lineage": "quarantine_lineage",
    "stop": "abort_run",
    "escalate_for_human_approval": None,
}

_NON_REVERSIBLE = {"quarantine_lineage", "stop"}


def plan_recovery(
    request: RecoveryRequest,
    classification: FailureClassification,
    evidence: list[NormalizedFailureEvidence],
    budget: RetryBudgetDecision,
    checkpoint: CheckpointRecoveryDecision,
) -> RecoveryRecommendation:
    signals = {signal for item in evidence for signal in item.signals}
    candidates = _candidate_actions(request, classification, signals, checkpoint)
    if not budget.allowed:
        candidates = ["stop"]
    requested = str(request.requested_action_kind or "")
    if requested and requested in candidates:
        action_kind = requested
    else:
        action_kind = candidates[0]
    registry_action = _REGISTRY_ACTION[action_kind]
    parameters: dict[str, object] = {
        "operation_id": request.operation_id,
        "failure_signature": classification.failure_signature,
    }
    if action_kind == "resume_verified_checkpoint":
        parameters.update(
            {
                "checkpoint_ref": checkpoint.checkpoint_ref,
                "resume_token_ref": checkpoint.resume_token_ref,
                "compatibility": checkpoint.compatibility,
            }
        )
    if action_kind == "rollback_verified_checkpoint":
        parameters.update(
            {
                "checkpoint_ref": request.constraints.get(
                    "verified_rollback_checkpoint_ref"
                ),
                "checkpoint_hash": request.constraints.get(
                    "verified_rollback_checkpoint_hash"
                ),
            }
        )
    if action_kind == "request_replacement_worker":
        parameters["reject_stale_results"] = True
        parameters["require_exclusive_new_lease"] = True
    if action_kind == "collect_more_evidence" and {
        "stale_result",
        "late_completion",
        "duplicate_ownership",
        "process_alive_unattached",
    }.intersection(signals):
        parameters["stale_completion_rejected"] = True
        parameters["exclusive_ownership_required"] = True

    payload = "|".join(
        [
            request.request_id,
            classification.classification_id,
            action_kind,
            registry_action or "advisory",
        ]
    )
    return RecoveryRecommendation(
        recommendation_id=f"rr-{hashlib.sha256(payload.encode()).hexdigest()[:16]}",
        action_kind=action_kind,
        registry_action=registry_action,
        rationale=_rationale(action_kind, classification.category),
        executable=registry_action is not None,
        reversible=action_kind not in _NON_REVERSIBLE,
        approval_required=action_kind
        in {
            "rollback_verified_checkpoint",
            "branch_new_experiment",
            "quarantine_lineage",
        },
        evidence_refs=list(classification.evidence_refs),
        parameters=parameters,
    )


def _candidate_actions(
    request: RecoveryRequest,
    classification: FailureClassification,
    signals: set[str],
    checkpoint: CheckpointRecoveryDecision,
) -> list[str]:
    category = classification.category
    if checkpoint.allowed and category in {
        "process_crash",
        "operator_interruption",
        "worker_lease_loss",
    }:
        return ["resume_verified_checkpoint", "restart_bounded_job"]
    if category == "transient_provider_outage":
        return ["retry_after_backoff"]
    if category == "transient_network_download_interruption":
        if "artifact_partial" in signals:
            return ["reacquire_partial_artifact", "retry_after_backoff"]
        return ["retry_after_backoff"]
    if category == "worker_lease_loss":
        uncertain = {
            "duplicate_ownership",
            "late_completion",
            "stale_result",
            "process_alive_unattached",
        }.intersection(signals)
        if uncertain:
            return ["collect_more_evidence", "stop"]
        return ["request_replacement_worker", "retry_after_backoff"]
    if category == "process_crash":
        return ["restart_bounded_job", "collect_more_evidence"]
    if category == "explicit_cancellation":
        return ["stop"]
    if category == "operator_interruption":
        return ["escalate_for_human_approval", "stop"]
    if category in {"out_of_memory", "resource_budget_exhaustion"}:
        return ["stop", "branch_new_experiment"]
    if category == "data_loader_failure":
        return ["retry_same_operation", "collect_more_evidence"]
    if category in {
        "malformed_or_contaminated_data",
        "tokenizer_incompatibility",
        "model_checkpoint_incompatibility",
        "resume_token_corruption",
        "invalid_configuration",
        "permanent_unsupported_capability",
    }:
        return ["stop", "branch_new_experiment"]
    if category in {"checkpoint_corruption", "deterministic_regression"}:
        if _verified_rollback_target(request):
            return ["rollback_verified_checkpoint", "stop"]
        return ["collect_more_evidence", "stop"]
    if category == "missing_checkpoint_evidence":
        return ["collect_more_evidence"]
    if category in {"missing_metrics", "incomplete_evaluation"}:
        return ["rerun_evaluation", "collect_more_evidence"]
    if category == "high_evaluation_variance":
        return ["rerun_evaluation", "collect_more_evidence"]
    if category == "replay_failure":
        return ["collect_more_evidence", "stop"]
    if category == "policy_or_approval_block":
        return ["escalate_for_human_approval"]
    if category == "poisoned_or_deprecated_lineage":
        return ["quarantine_lineage", "stop"]
    if category == "storage_integrity_failure":
        return ["reacquire_partial_artifact", "collect_more_evidence"]
    if category == "state_persistence_failure":
        return ["retry_same_operation", "stop"]
    return ["collect_more_evidence"]


def _verified_rollback_target(request: RecoveryRequest) -> bool:
    return bool(request.constraints.get("verified_rollback_checkpoint_ref")) and bool(
        request.constraints.get("verified_rollback_checkpoint_hash")
    )


def _rationale(action_kind: str, category: str) -> str:
    return (
        f"Recommend {action_kind} for explicit {category} evidence; "
        "authorization and execution remain separate."
    )
