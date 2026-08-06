"""Composed bounded recovery decision service."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass, field

from hephaestus.config_loader import ConfigError
from hephaestus.policy.recovery_policy import RecoveryPolicy
from hephaestus.policy.stage_policy import StagePolicy
from hephaestus.recovery.budgets import decide_backoff, evaluate_retry_budget
from hephaestus.recovery.checkpoint import validate_checkpoint_recovery
from hephaestus.recovery.classification import classify_failure
from hephaestus.recovery.models import (
    RECOVERY_ACTION_KINDS,
    BackoffDecision,
    CheckpointRecoveryDecision,
    FailureClassification,
    RecoveryDecision,
    RecoveryRecommendation,
    RecoveryRequest,
    RetryBudgetDecision,
)
from hephaestus.recovery.normalization import normalize_failure_evidence
from hephaestus.recovery.planning import plan_recovery
from hephaestus.recovery.store import RecoveryAttemptStore
from hephaestus.schemas.contract_common import ContractIssue


@dataclass(slots=True)
class BoundedRecoveryService:
    attempt_store: RecoveryAttemptStore
    policy: RecoveryPolicy = field(default_factory=RecoveryPolicy)
    stage_policy: StagePolicy = field(default_factory=StagePolicy)

    def decide(self, request: RecoveryRequest) -> RecoveryDecision:
        if not isinstance(request, RecoveryRequest):
            return self._invalid_request(request)
        raw_evidence = [
            deepcopy(item) for item in request.evidence if isinstance(item, dict)
        ]
        malformed_count = len(request.evidence) - len(raw_evidence)
        normalized = normalize_failure_evidence(raw_evidence)
        classification = classify_failure(request, normalized)
        attempts = self.attempt_store.list_attempts()
        budget = evaluate_retry_budget(request, classification, attempts, self.policy)
        backoff = decide_backoff(request, classification, budget, self.policy)
        checkpoint = validate_checkpoint_recovery(request, normalized)
        recommendation = plan_recovery(
            request,
            classification,
            normalized,
            budget,
            checkpoint,
        )

        issues = list(classification.issues)
        if malformed_count:
            issues.append(
                ContractIssue(
                    code="malformed_recovery_evidence",
                    category="invalid_request",
                    message=f"{malformed_count} recovery evidence records were not mappings.",
                    blocking=False,
                )
            )
        if not budget.allowed:
            issues.append(
                ContractIssue(
                    code="recovery_retry_budget_exhausted",
                    category="budget_exceeded",
                    message="Recovery retry or cost policy blocks another retry.",
                    blocking=True,
                    evidence_refs=classification.evidence_refs,
                    metadata={"reasons": budget.reasons},
                )
            )

        requested_action = str(request.requested_action_kind or "")
        requested_action_blocked = False
        if requested_action:
            if requested_action not in RECOVERY_ACTION_KINDS:
                requested_action_blocked = True
                issues.append(
                    ContractIssue(
                        code="unknown_recovery_action",
                        category="invalid_request",
                        message=f"Recovery action {requested_action!r} is not recognized.",
                        blocking=True,
                    )
                )
            elif recommendation.action_kind != requested_action:
                requested_action_blocked = True
                issues.append(
                    ContractIssue(
                        code="requested_recovery_action_ineligible",
                        category="policy_blocked",
                        message=(
                            f"Requested action {requested_action!r} is not eligible for the "
                            f"classified failure {classification.category!r}."
                        ),
                        blocking=True,
                        evidence_refs=classification.evidence_refs,
                    )
                )

        checkpoint_relevant = (
            recommendation.action_kind == "resume_verified_checkpoint"
            or requested_action == "resume_verified_checkpoint"
        )
        if checkpoint_relevant and not checkpoint.allowed:
            issues.extend(checkpoint.issues)

        trust_level = _trust_level(normalized, request)
        stage_actions, stage_issue = self._stage_actions(request.stage_name)
        if stage_issue is not None:
            issues.append(stage_issue)
        approval_status, approval_ref = _approval_for_action(
            request,
            recommendation.registry_action,
        )
        recommendation.approval_ref = approval_ref
        boundary = self.policy.assess_registered_action(
            recommendation.registry_action,
            stage_name=request.stage_name,
            trust_level=trust_level,
            approval_status=approval_status,
            stage_allowed_actions=stage_actions,
        )
        recommendation.approval_required = bool(boundary["requires_approval"])

        status = self._status(
            classification,
            recommendation,
            budget,
            checkpoint,
            boundary,
            requested_action_blocked,
        )
        if status == "approval_required":
            issues.append(
                ContractIssue(
                    code="recovery_approval_required",
                    category="approval_required",
                    message="Existing approval governance requires matching approval evidence before execution.",
                    blocking=True,
                    evidence_refs=classification.evidence_refs,
                    metadata={"registry_action": recommendation.registry_action},
                )
            )
        if status == "blocked" and not any(issue.blocking for issue in issues):
            issues.append(
                ContractIssue(
                    code="recovery_policy_blocked",
                    category="policy_blocked",
                    message="Recovery eligibility policy blocks execution.",
                    blocking=True,
                    metadata={"reasons": boundary.get("reasons", [])},
                )
            )

        attempt_id = _attempt_id(request, classification, recommendation)
        decision_id = f"rd-{hashlib.sha256((request.request_id + attempt_id).encode()).hexdigest()[:16]}"
        return RecoveryDecision(
            decision_id=decision_id,
            request_id=request.request_id,
            run_id=request.run_id,
            experiment_id=request.experiment_id,
            lineage_id=request.lineage_id,
            stage_name=request.stage_name,
            status=status,
            classification=classification,
            recommendation=recommendation,
            budget=budget,
            backoff=backoff,
            checkpoint=checkpoint if checkpoint_relevant else None,
            attempt_id=attempt_id,
            evidence_refs=sorted({item.source_ref for item in normalized}),
            issues=_deduplicate_issues(issues),
            metadata={
                "deterministic_decision": True,
                "diagnosis_authority_separate": True,
                "recommendation_authority_separate": True,
                "authorization_authority_separate": True,
                "execution_authority_separate": True,
                "action_executed": False,
                "judge_verdict_mutated": False,
                "checkpoint_promoted": False,
                "operation_id": request.operation_id,
                "budget_window_id": str(
                    request.constraints.get("budget_window_id") or "default"
                ),
                "estimated_recovery_cost": _safe_float(
                    request.constraints.get("estimated_recovery_cost"), 0.0
                ),
                "request_progress_observed": bool(
                    request.constraints.get("progress_observed", False)
                ),
                "approval_status": approval_status,
                "approval_ref": approval_ref,
                "trust_level": trust_level,
                "stage_allowed_actions": sorted(stage_actions)
                if stage_actions is not None
                else None,
                "action_boundary": boundary,
                "normalized_evidence": [item.to_dict() for item in normalized],
            },
        )

    def _status(
        self,
        classification: FailureClassification,
        recommendation: RecoveryRecommendation,
        budget: RetryBudgetDecision,
        checkpoint: CheckpointRecoveryDecision,
        boundary: dict[str, object],
        requested_action_blocked: bool,
    ) -> str:
        if requested_action_blocked:
            return "blocked"
        if classification.category == "unknown_inconclusive" or (
            classification.confidence < self.policy.minimum_recovery_confidence
        ):
            return "inconclusive"
        if (
            recommendation.action_kind == "resume_verified_checkpoint"
            and not checkpoint.allowed
        ):
            return "blocked"
        if recommendation.registry_action is None:
            return "approval_required"
        if bool(boundary["requires_approval"]) and not bool(boundary["allowed"]):
            return "approval_required"
        if not bool(boundary["known_action"]) or bool(boundary["forbidden"]):
            return "blocked"
        if not bool(boundary.get("stage_allowed", True)):
            return "blocked"
        safety_action = recommendation.registry_action in {
            "abort_run",
            "request_recheck",
        }
        approved_action = bool(boundary["requires_approval"]) and bool(
            boundary["allowed"]
        )
        if not budget.allowed and recommendation.registry_action != "abort_run":
            return "blocked"
        if (
            not classification.safe_to_automate
            and not safety_action
            and not approved_action
        ):
            return "blocked"
        if classification.contradicting_evidence_refs and not approved_action:
            return "inconclusive"
        return "eligible"

    def _stage_actions(
        self, stage_name: str
    ) -> tuple[set[str] | None, ContractIssue | None]:
        try:
            profile = self.stage_policy.resolve(stage_name)
        except (ConfigError, OSError, ValueError) as exc:
            return set(), ContractIssue(
                code="recovery_stage_policy_unavailable",
                category="policy_blocked",
                message=f"Stage policy could not be resolved: {type(exc).__name__}.",
                blocking=False,
            )
        return set(profile.allowed_next_actions), None

    def _invalid_request(self, request: object) -> RecoveryDecision:
        request_id = str(getattr(request, "request_id", "invalid-recovery-request"))
        classification = FailureClassification(
            classification_id="rc-invalid",
            category="unknown_inconclusive",
            likely_failure_domain="inconclusive",
            confidence=0.0,
            retryability="unknown",
            requires_new_evidence=True,
            issues=[
                ContractIssue(
                    code="invalid_recovery_request",
                    category="invalid_request",
                    message="decide requires RecoveryRequest.",
                    blocking=True,
                )
            ],
        )
        recommendation = RecoveryRecommendation(
            recommendation_id="rr-invalid",
            action_kind="collect_more_evidence",
            registry_action="request_recheck",
            rationale="A valid RecoveryRequest is required.",
            executable=False,
            reversible=True,
            approval_required=False,
        )
        return RecoveryDecision(
            decision_id="rd-invalid",
            request_id=request_id,
            run_id=str(getattr(request, "run_id", "")),
            experiment_id=str(getattr(request, "experiment_id", "")),
            lineage_id=str(getattr(request, "lineage_id", "")),
            stage_name=str(getattr(request, "stage_name", "")),
            status="inconclusive",
            classification=classification,
            recommendation=recommendation,
            budget=RetryBudgetDecision(allowed=False, reasons=["invalid_request"]),
            backoff=BackoffDecision(False, 0, "default", 0),
            checkpoint=None,
            attempt_id="ra-invalid",
            issues=list(classification.issues),
            metadata={"action_executed": False},
        )


def _approval_for_action(
    request: RecoveryRequest, registry_action: str | None
) -> tuple[str, str | None]:
    if registry_action is None:
        return "missing", None
    for item in reversed(request.approval_evidence):
        metadata = item.get("metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        action = str(
            item.get("proposed_action") or metadata.get("proposed_action") or ""
        )
        status = str(item.get("status") or item.get("outcome") or "").lower()
        effect = str(item.get("effect_on_action") or "")
        if action != registry_action or status not in {"approved", "override_approved"}:
            continue
        if effect and effect != "execute_requested_action":
            continue
        if item.get("run_id") and str(item["run_id"]) != request.run_id:
            continue
        if item.get("lineage_id") and str(item["lineage_id"]) != request.lineage_id:
            continue
        ref = (
            str(
                item.get("source_ref")
                or item.get("decision_event_id")
                or item.get("request_id")
                or ""
            )
            or None
        )
        return status, ref
    return "missing", None


def _trust_level(normalized: list, request: RecoveryRequest) -> str:
    for item in normalized:
        if item.evidence_kind in {"lineage", "lineage_state"}:
            value = str(item.payload.get("trust_level") or "").strip()
            if value:
                return value
    return str(request.constraints.get("lineage_trust_level") or "unknown")


def _attempt_id(
    request: RecoveryRequest,
    classification: FailureClassification,
    recommendation: RecoveryRecommendation,
) -> str:
    payload = (
        f"{request.request_id}|{request.operation_id}|"
        f"{classification.failure_signature}|{classification.evidence_fingerprint}|"
        f"{recommendation.recommendation_id}"
    )
    return f"ra-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def _deduplicate_issues(issues: list[ContractIssue]) -> list[ContractIssue]:
    unique: dict[tuple[str, str], ContractIssue] = {}
    for issue in issues:
        unique[(issue.code, issue.message)] = issue
    return [unique[key] for key in sorted(unique)]


def _safe_float(value: object, default: float) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return default
