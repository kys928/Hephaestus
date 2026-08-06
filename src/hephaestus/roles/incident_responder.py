"""Thin role boundary for recovery decisions and separately authorized execution."""

from __future__ import annotations

from dataclasses import dataclass

from hephaestus.recovery.controller import RecoveryController
from hephaestus.recovery.models import (
    RecoveryDecision,
    RecoveryExecutionResult,
    RecoveryRequest,
)
from hephaestus.recovery.service import BoundedRecoveryService


@dataclass(slots=True)
class IncidentResponderRole:
    service: BoundedRecoveryService
    controller: RecoveryController | None = None

    def assess(self, request: RecoveryRequest) -> RecoveryDecision:
        """Classify and recommend; never execute implicitly."""

        return self.service.decide(request)

    def execute_approved(self, decision: RecoveryDecision) -> RecoveryExecutionResult:
        """Execute only through an explicitly injected bounded controller."""

        if self.controller is None:
            raise RuntimeError("recovery controller is not configured")
        return self.controller.execute(decision)
