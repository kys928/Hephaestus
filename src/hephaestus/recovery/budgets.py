"""Retry budgets, deterministic backoff, and loop prevention."""

from __future__ import annotations

import hashlib

from hephaestus.policy.recovery_policy import RecoveryPolicy
from hephaestus.recovery.models import (
    BackoffDecision,
    FailureClassification,
    RecoveryAttempt,
    RecoveryRequest,
    RetryBudgetDecision,
)

_CONSUMING_STATUSES = {"executing", "succeeded", "failed"}


def evaluate_retry_budget(
    request: RecoveryRequest,
    classification: FailureClassification,
    attempts: list[RecoveryAttempt],
    policy: RecoveryPolicy,
) -> RetryBudgetDecision:
    relevant = [item for item in attempts if item.status in _CONSUMING_STATUSES]
    since_progress = _since_last_progress(relevant, request.operation_id)
    window_id = str(request.constraints.get("budget_window_id") or "default")
    counts = {
        "operation": sum(
            item.operation_id == request.operation_id for item in since_progress
        ),
        "failure_signature": sum(
            item.failure_signature == classification.failure_signature
            for item in since_progress
        ),
        "run": sum(item.run_id == request.run_id for item in relevant),
        "experiment": sum(
            item.experiment_id == request.experiment_id for item in relevant
        ),
        "lineage": sum(item.lineage_id == request.lineage_id for item in relevant),
        "global_window": sum(
            str(item.metadata.get("budget_window_id") or "default") == window_id
            for item in relevant
        ),
    }
    limits = policy.limits()
    exhausted = sorted(
        scope for scope, count in counts.items() if count >= limits[scope]
    )
    identical = sum(
        item.failure_signature == classification.failure_signature
        and item.evidence_fingerprint == classification.evidence_fingerprint
        and not item.progress_observed
        for item in since_progress
    )
    failed_same_action = _failed_same_action_count(
        since_progress,
        classification.failure_signature,
        request.requested_action_kind,
    )
    cumulative_cost = sum(
        item.cost for item in relevant if item.lineage_id == request.lineage_id
    )
    requested_cost = _safe_float(
        request.constraints.get("estimated_recovery_cost"), 0.0
    )
    reasons: list[str] = []
    if exhausted:
        reasons.append("retry_budget_exhausted:" + ",".join(exhausted))
    if identical >= policy.identical_evidence_retry_limit:
        reasons.append("identical_evidence_without_progress")
    if failed_same_action >= policy.failed_action_retry_limit:
        reasons.append("same_recovery_action_repeatedly_failed")
    if _oscillating(since_progress, classification.failure_signature):
        reasons.append("recovery_action_oscillation")
    if cumulative_cost + requested_cost > policy.cumulative_cost_limit:
        reasons.append("cumulative_recovery_cost_exceeded")
    return RetryBudgetDecision(
        allowed=not reasons,
        counts={key: int(value) for key, value in counts.items()},
        limits=limits,
        exhausted_scopes=exhausted,
        identical_evidence_attempts=identical,
        cumulative_cost=round(cumulative_cost, 6),
        reasons=reasons,
    )


def decide_backoff(
    request: RecoveryRequest,
    classification: FailureClassification,
    budget: RetryBudgetDecision,
    policy: RecoveryPolicy,
) -> BackoffDecision:
    category = classification.category
    if category == "transient_provider_outage":
        kind = "provider"
    elif category == "worker_lease_loss":
        kind = "worker"
    elif category in {"storage_integrity_failure", "state_persistence_failure"}:
        kind = "storage"
    elif category in {"process_crash", "out_of_memory", "data_loader_failure"}:
        kind = "training"
    else:
        kind = "default"
    required = category in {
        "transient_provider_outage",
        "transient_network_download_interruption",
        "worker_lease_loss",
        "process_crash",
        "data_loader_failure",
        "storage_integrity_failure",
        "state_persistence_failure",
    }
    attempt_index = budget.counts.get("failure_signature", 0)
    base = policy.backoff_bases[kind]
    bounded = min(policy.backoff_maximum_seconds, base * (2**attempt_index))
    jitter = 0
    if required and policy.deterministic_jitter_max_seconds > 0:
        digest = hashlib.sha256(
            f"{request.request_id}|{classification.failure_signature}|{attempt_index}".encode()
        ).digest()
        jitter = int.from_bytes(digest[:2], "big") % (
            policy.deterministic_jitter_max_seconds + 1
        )
    return BackoffDecision(
        required=required,
        delay_seconds=min(policy.backoff_maximum_seconds, bounded + jitter)
        if required
        else 0,
        policy_kind=kind,
        attempt_index=attempt_index,
        deterministic_jitter_seconds=jitter,
        maximum_delay_seconds=policy.backoff_maximum_seconds,
        reset_after_progress=True,
    )


def _since_last_progress(
    attempts: list[RecoveryAttempt], operation_id: str
) -> list[RecoveryAttempt]:
    relevant = [item for item in attempts if item.operation_id == operation_id]
    last_progress = -1
    for index, item in enumerate(relevant):
        if item.progress_observed:
            last_progress = index
    return relevant[last_progress + 1 :]


def _failed_same_action_count(
    attempts: list[RecoveryAttempt],
    failure_signature: str,
    requested_action: str | None,
) -> int:
    rows = [
        item
        for item in attempts
        if item.failure_signature == failure_signature and item.status == "failed"
    ]
    if requested_action:
        rows = [item for item in rows if item.action_kind == requested_action]
    if not rows:
        return 0
    latest_action = rows[-1].action_kind
    return sum(item.action_kind == latest_action for item in rows)


def _oscillating(attempts: list[RecoveryAttempt], failure_signature: str) -> bool:
    actions = [
        item.action_kind
        for item in attempts
        if item.failure_signature == failure_signature and item.status == "failed"
    ][-4:]
    return (
        len(actions) == 4
        and actions[0] == actions[2]
        and actions[1] == actions[3]
        and actions[0] != actions[1]
    )


def _safe_float(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, parsed)
