"""Injected recovery-attempt persistence boundary and deterministic fake."""

from __future__ import annotations

import threading
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from hephaestus.recovery.models import RecoveryAttempt


class RecoveryAttemptConflict(RuntimeError):
    pass


@runtime_checkable
class RecoveryAttemptStore(Protocol):
    def get(self, attempt_id: str) -> RecoveryAttempt | None: ...

    def list_attempts(self) -> list[RecoveryAttempt]: ...

    def record(self, attempt: RecoveryAttempt) -> RecoveryAttempt: ...

    def update(self, attempt: RecoveryAttempt) -> RecoveryAttempt: ...


@dataclass(slots=True)
class InMemoryRecoveryAttemptStore:
    """Thread-safe deterministic fixture; it does not claim durable guarantees."""

    _attempts: dict[str, RecoveryAttempt] = field(default_factory=dict, init=False)
    _order: list[str] = field(default_factory=list, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def get(self, attempt_id: str) -> RecoveryAttempt | None:
        with self._lock:
            attempt = self._attempts.get(attempt_id)
            return (
                None
                if attempt is None
                else RecoveryAttempt.from_dict(deepcopy(attempt.to_dict()))
            )

    def list_attempts(self) -> list[RecoveryAttempt]:
        with self._lock:
            return [
                RecoveryAttempt.from_dict(deepcopy(self._attempts[item].to_dict()))
                for item in self._order
            ]

    def record(self, attempt: RecoveryAttempt) -> RecoveryAttempt:
        candidate = RecoveryAttempt.from_dict(deepcopy(attempt.to_dict()))
        with self._lock:
            existing = self._attempts.get(candidate.attempt_id)
            if existing is not None:
                if existing.to_dict() != candidate.to_dict():
                    raise RecoveryAttemptConflict(
                        f"attempt ID {candidate.attempt_id!r} already has different content"
                    )
                return RecoveryAttempt.from_dict(deepcopy(existing.to_dict()))
            self._attempts[candidate.attempt_id] = candidate
            self._order.append(candidate.attempt_id)
            return RecoveryAttempt.from_dict(deepcopy(candidate.to_dict()))

    def update(self, attempt: RecoveryAttempt) -> RecoveryAttempt:
        candidate = RecoveryAttempt.from_dict(deepcopy(attempt.to_dict()))
        with self._lock:
            if candidate.attempt_id not in self._attempts:
                self._order.append(candidate.attempt_id)
            self._attempts[candidate.attempt_id] = candidate
            return RecoveryAttempt.from_dict(deepcopy(candidate.to_dict()))
