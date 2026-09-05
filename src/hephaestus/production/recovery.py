"""Automatic infrastructure-only retry policy for the production loop.

Recovery in this module is intentionally narrower than scientific recovery: it
may retry transport/scheduler/bootstrap failures only when there is no evidence
that an optimizer step or irreversible governed action occurred.  It never
changes model/data/optimizer/evaluation variables.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

from hephaestus.storage.base import StateRepository

T = TypeVar("T")
INFRA_RECOVERY = "production_infrastructure_recovery"

_SAFE_CODES = frozenset(
    {
        "capacity_unavailable",
        "scheduler_unavailable",
        "provider_rate_limited",
        "transport_interrupted",
        "pod_bootstrap_failed",
        "worker_lost_before_progress",
        "stale_execution_sentinel",
        "stale_execution_log",
        "admission_ceiling_before_step_one",
        "temporary_object_store_unavailable",
    }
)


class RecoverableInfrastructureError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        optimizer_steps: int = 0,
        checkpoint_created: bool = False,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.optimizer_steps = int(optimizer_steps)
        self.checkpoint_created = bool(checkpoint_created)
        self.details = details or {}


@dataclass(slots=True)
class InfrastructureRecoveryController:
    repository: StateRepository
    maximum_attempts: int = 5
    retry_codes: frozenset[str] = field(default_factory=lambda: _SAFE_CODES)

    def _operation_attempts(self, operation_id: str) -> list[dict[str, object]]:
        return [
            row
            for row in self.repository.all(INFRA_RECOVERY)
            if row.get("operation_id") == operation_id
        ]

    def run(
        self,
        operation_id: str,
        operation: Callable[[int], T],
        *,
        on_retry: Callable[[RecoverableInfrastructureError, int], None] | None = None,
    ) -> T:
        """Execute and automatically retry safe infrastructure failures.

        The callback receives the failure plus the next attempt number and may
        perform idempotent cleanup such as archiving a stale terminal sentinel.
        Scientific configuration changes are not permitted by this boundary.
        """
        history = self._operation_attempts(operation_id)
        start_attempt = 1 + max([int(row.get("attempt", 0)) for row in history] or [0])
        last: RecoverableInfrastructureError | None = None
        for attempt in range(start_attempt, self.maximum_attempts + 1):
            try:
                value = operation(attempt)
            except RecoverableInfrastructureError as exc:
                last = exc
                safe = (
                    exc.code in self.retry_codes
                    and exc.optimizer_steps == 0
                    and not exc.checkpoint_created
                )
                record = {
                    "record_id": "infra-recovery-" + hashlib.sha256(
                        json.dumps([operation_id, attempt, exc.code, exc.details], sort_keys=True, default=str).encode()
                    ).hexdigest()[:20],
                    "operation_id": operation_id,
                    "attempt": attempt,
                    "status": "retrying" if safe and attempt < self.maximum_attempts else "blocked",
                    "failure_code": exc.code,
                    "message": str(exc),
                    "optimizer_steps": exc.optimizer_steps,
                    "checkpoint_created": exc.checkpoint_created,
                    "scientific_variables_changed": False,
                    "details": dict(exc.details),
                }
                self.repository.append(INFRA_RECOVERY, record)
                if not safe or attempt >= self.maximum_attempts:
                    raise
                if on_retry is not None:
                    on_retry(exc, attempt + 1)
                continue
            self.repository.append(
                INFRA_RECOVERY,
                {
                    "record_id": "infra-success-" + hashlib.sha256(f"{operation_id}:{attempt}".encode()).hexdigest()[:20],
                    "operation_id": operation_id,
                    "attempt": attempt,
                    "status": "completed",
                    "scientific_variables_changed": False,
                },
            )
            return value
        assert last is not None
        raise last

    @staticmethod
    def classify_message(message: str, *, optimizer_steps: int = 0, checkpoint_created: bool = False) -> RecoverableInfrastructureError | None:
        """Normalize common provider/runtime messages into safe retry classes."""
        text = str(message).casefold()
        mapping = (
            (("capacity", "no available", "stock"), "capacity_unavailable"),
            (("rate limit", "429"), "provider_rate_limited"),
            (("timeout", "connection reset", "temporarily unavailable"), "transport_interrupted"),
            (("stale sentinel", "repo_sha mismatch"), "stale_execution_sentinel"),
            (("bootstrap",), "pod_bootstrap_failed"),
            (("max_total_tokens", "admission ceiling"), "admission_ceiling_before_step_one"),
            (("worker lost", "process_exit_evidence_missing"), "worker_lost_before_progress"),
        )
        for needles, code in mapping:
            if any(needle in text for needle in needles):
                return RecoverableInfrastructureError(
                    code,
                    message,
                    optimizer_steps=optimizer_steps,
                    checkpoint_created=checkpoint_created,
                )
        return None
