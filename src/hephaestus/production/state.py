"""Durable control records used by the production loop.

These adapters deliberately persist only compact JSON-safe control state. Heavy
artifacts remain behind the existing artifact-store boundaries.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from hephaestus.recovery.models import RecoveryAttempt
from hephaestus.recovery.store import RecoveryAttemptConflict, RecoveryAttemptStore
from hephaestus.storage.base import StateRepository

RECOVERY_ATTEMPTS = "production_recovery_attempts"
LOOP_STATES = "production_loop_states"
LOOP_EVENTS = "production_loop_events"
ACTION_EXECUTIONS = "production_action_executions"
INTEGRATION_RECORDS = "production_integration_records"


@dataclass(slots=True)
class RepositoryIntegrationRecordSink:
    repository: StateRepository

    def append(self, kind: str, payload: dict[str, object]) -> None:
        self.repository.append(
            INTEGRATION_RECORDS,
            {"kind": str(kind), "payload": dict(payload)},
        )


@dataclass(slots=True)
class DurableRecoveryAttemptStore(RecoveryAttemptStore):
    """Append-only recovery store backed by the configured StateRepository."""

    repository: StateRepository

    def get(self, attempt_id: str) -> RecoveryAttempt | None:
        row = self.repository.get_latest(RECOVERY_ATTEMPTS, "attempt_id", attempt_id)
        if row is None:
            return None
        payload = dict(row.get("attempt", {})) if isinstance(row.get("attempt"), dict) else dict(row)
        return RecoveryAttempt.from_dict(payload)

    def list_attempts(self) -> list[RecoveryAttempt]:
        latest: dict[str, dict[str, object]] = {}
        for row in self.repository.all(RECOVERY_ATTEMPTS):
            attempt_id = str(row.get("attempt_id", ""))
            if attempt_id:
                latest[attempt_id] = row
        attempts: list[RecoveryAttempt] = []
        for attempt_id in sorted(latest):
            row = latest[attempt_id]
            payload = dict(row.get("attempt", {})) if isinstance(row.get("attempt"), dict) else dict(row)
            attempts.append(RecoveryAttempt.from_dict(payload))
        return attempts

    def record(self, attempt: RecoveryAttempt) -> RecoveryAttempt:
        existing = self.get(attempt.attempt_id)
        if existing is not None:
            if existing.to_dict() != attempt.to_dict():
                raise RecoveryAttemptConflict(
                    f"attempt ID {attempt.attempt_id!r} already has different content"
                )
            return existing
        self.repository.append(
            RECOVERY_ATTEMPTS,
            {"attempt_id": attempt.attempt_id, "version": 1, "attempt": attempt.to_dict()},
        )
        return RecoveryAttempt.from_dict(attempt.to_dict())

    def update(self, attempt: RecoveryAttempt) -> RecoveryAttempt:
        versions = [
            int(row.get("version", 0))
            for row in self.repository.all(RECOVERY_ATTEMPTS)
            if row.get("attempt_id") == attempt.attempt_id
        ]
        self.repository.append(
            RECOVERY_ATTEMPTS,
            {
                "attempt_id": attempt.attempt_id,
                "version": (max(versions) if versions else 0) + 1,
                "attempt": attempt.to_dict(),
            },
        )
        return RecoveryAttempt.from_dict(attempt.to_dict())


@dataclass(slots=True)
class ProductionLoopState:
    program_id: str
    lineage_id: str
    stage_name: str
    status: str = "pending"
    cycle_index: int = 0
    latest_run_id: str | None = None
    latest_experiment_id: str | None = None
    latest_comparison_ref: str | None = None
    latest_judge_action: str | None = None
    stop_reason: str | None = None
    recovery_attempts: int = 0
    completed_cycles: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ProductionLoopState":
        return cls(
            program_id=str(payload.get("program_id", "")),
            lineage_id=str(payload.get("lineage_id", "")),
            stage_name=str(payload.get("stage_name", "")),
            status=str(payload.get("status", "pending")),
            cycle_index=int(payload.get("cycle_index", 0)),
            latest_run_id=str(payload["latest_run_id"]) if payload.get("latest_run_id") else None,
            latest_experiment_id=str(payload["latest_experiment_id"]) if payload.get("latest_experiment_id") else None,
            latest_comparison_ref=str(payload["latest_comparison_ref"]) if payload.get("latest_comparison_ref") else None,
            latest_judge_action=str(payload["latest_judge_action"]) if payload.get("latest_judge_action") else None,
            stop_reason=str(payload["stop_reason"]) if payload.get("stop_reason") else None,
            recovery_attempts=int(payload.get("recovery_attempts", 0)),
            completed_cycles=[str(item) for item in payload.get("completed_cycles", [])],
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), dict) else {},
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ProductionLoopStateStore:
    repository: StateRepository

    def get(self, program_id: str) -> ProductionLoopState | None:
        row = self.repository.get_latest(LOOP_STATES, "program_id", program_id)
        return None if row is None else ProductionLoopState.from_dict(row)

    def save(self, state: ProductionLoopState) -> None:
        self.repository.append(LOOP_STATES, state.to_dict())

    def event(self, program_id: str, kind: str, payload: dict[str, object]) -> None:
        rows = [row for row in self.repository.all(LOOP_EVENTS) if row.get("program_id") == program_id]
        self.repository.append(
            LOOP_EVENTS,
            {
                "program_id": program_id,
                "sequence": len(rows) + 1,
                "kind": kind,
                "payload": payload,
            },
        )
