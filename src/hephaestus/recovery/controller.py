"""Narrow, injected, allowlisted, idempotent recovery action controller."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from hephaestus.policy.action_registry import evaluate_action_boundary
from hephaestus.recovery.models import (
    RecoveryAttempt,
    RecoveryDecision,
    RecoveryExecutionResult,
)
from hephaestus.recovery.store import RecoveryAttemptStore
from hephaestus.schemas.contract_common import ContractIssue


@runtime_checkable
class RecoveryActionExecutor(Protocol):
    def execute(
        self,
        action_kind: str,
        parameters: dict[str, object],
        attempt_id: str,
    ) -> str | None: ...


@dataclass(slots=True)
class FakeRecoveryActionExecutor:
    """Deterministic fixture executor; no external work is performed."""

    result_refs: dict[str, str | None] = field(default_factory=dict)
    failing_actions: set[str] = field(default_factory=set)
    calls: list[tuple[str, str]] = field(default_factory=list)

    def execute(
        self,
        action_kind: str,
        parameters: dict[str, object],
        attempt_id: str,
    ) -> str | None:
        self.calls.append((attempt_id, action_kind))
        if action_kind in self.failing_actions:
            raise RuntimeError("deterministic fake failure")
        return self.result_refs.get(action_kind, f"fake://recovery/{attempt_id}")


@dataclass(slots=True)
class RecoveryController:
    attempt_store: RecoveryAttemptStore
    executors: dict[str, RecoveryActionExecutor]
    allowlisted_actions: set[str]

    def execute(self, decision: RecoveryDecision) -> RecoveryExecutionResult:
        recommendation = decision.recommendation
        if decision.status != "eligible":
            return self._blocked(
                decision,
                "recovery_decision_not_eligible",
                f"Recovery decision status {decision.status!r} does not authorize execution.",
            )
        if not recommendation.executable or recommendation.registry_action is None:
            return self._blocked(
                decision,
                "recovery_action_not_executable",
                "Recommendation is advisory and has no executable registered action.",
            )
        if recommendation.action_kind not in self.allowlisted_actions:
            return self._blocked(
                decision,
                "recovery_action_not_allowlisted",
                f"Recovery action {recommendation.action_kind!r} is not allowlisted.",
            )
        executor = self.executors.get(recommendation.action_kind)
        if executor is None:
            return self._blocked(
                decision,
                "recovery_executor_missing",
                "No injected executor exists for the allowlisted recovery action.",
            )

        approval_status = str(decision.metadata.get("approval_status") or "missing")
        boundary = evaluate_action_boundary(
            recommendation.registry_action,
            context={"approval_status": approval_status},
        )
        if not bool(boundary["allowed"]):
            return self._blocked(
                decision,
                "recovery_action_boundary_blocked",
                "Existing action-boundary governance refused execution.",
                metadata={"action_boundary": boundary},
            )
        if recommendation.approval_required and not recommendation.approval_ref:
            return self._blocked(
                decision,
                "recovery_approval_evidence_missing",
                "High-impact recovery requires a matching approval evidence reference.",
            )

        existing = self.attempt_store.get(decision.attempt_id)
        if existing is not None:
            return RecoveryExecutionResult(
                attempt_id=existing.attempt_id,
                status=f"already_{existing.status}",
                action_kind=existing.action_kind,
                result_ref=existing.result_ref,
                metadata={"idempotent_replay": True},
            )

        attempt = RecoveryAttempt(
            attempt_id=decision.attempt_id,
            request_id=decision.request_id,
            run_id=decision.run_id,
            experiment_id=decision.experiment_id,
            lineage_id=decision.lineage_id,
            operation_id=str(decision.metadata.get("operation_id") or ""),
            failure_signature=decision.classification.failure_signature,
            evidence_fingerprint=decision.classification.evidence_fingerprint,
            action_kind=recommendation.action_kind,
            registry_action=recommendation.registry_action,
            status="executing",
            input_evidence_refs=list(decision.evidence_refs),
            cost=float(decision.metadata.get("estimated_recovery_cost") or 0.0),
            progress_observed=bool(
                decision.metadata.get("request_progress_observed", False)
            ),
            metadata={
                "decision_id": decision.decision_id,
                "budget_window_id": decision.metadata.get(
                    "budget_window_id", "default"
                ),
                "approval_ref": recommendation.approval_ref,
                "idempotent": True,
            },
        )
        self.attempt_store.record(attempt)
        try:
            result_ref = executor.execute(
                recommendation.action_kind,
                dict(recommendation.parameters),
                decision.attempt_id,
            )
        except Exception as exc:  # noqa: BLE001 - injected executor trust boundary
            attempt.status = "failed"
            attempt.metadata["error_type"] = type(exc).__name__
            self.attempt_store.update(attempt)
            return RecoveryExecutionResult(
                attempt_id=attempt.attempt_id,
                status="failed",
                action_kind=attempt.action_kind,
                issues=[
                    ContractIssue(
                        code="recovery_executor_failed",
                        category="runtime_failure",
                        message=f"Injected recovery executor failed: {type(exc).__name__}.",
                        retryable=True,
                        blocking=True,
                        evidence_refs=list(decision.evidence_refs),
                    )
                ],
                metadata={"handler_error_redacted": True},
            )
        attempt.status = "succeeded"
        attempt.result_ref = result_ref
        self.attempt_store.update(attempt)
        return RecoveryExecutionResult(
            attempt_id=attempt.attempt_id,
            status="succeeded",
            action_kind=attempt.action_kind,
            result_ref=result_ref,
            metadata={
                "idempotent": True,
                "judge_verdict_mutated": False,
                "checkpoint_promoted": False,
            },
        )

    @staticmethod
    def _blocked(
        decision: RecoveryDecision,
        code: str,
        message: str,
        metadata: dict[str, object] | None = None,
    ) -> RecoveryExecutionResult:
        return RecoveryExecutionResult(
            attempt_id=decision.attempt_id,
            status="blocked",
            action_kind=decision.recommendation.action_kind,
            issues=[
                ContractIssue(
                    code=code,
                    category="policy_blocked",
                    message=message,
                    blocking=True,
                    evidence_refs=list(decision.evidence_refs),
                )
            ],
            metadata=metadata or {},
        )
