"""Runtime Monitor role contract stub."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RuntimeOutcome(str, Enum):
    """Finite runtime outcome categories for Stage 1 architecture boundaries."""

    HEALTHY = "healthy"
    SOFT_SUSPICION = "soft_suspicion"
    WASTE_STOP = "waste_stop"
    HARD_ABORT = "hard_abort"


@dataclass(slots=True)
class RuntimeMonitorResult:
    """Typed monitor output for explicit control-plane handoff."""

    outcome: RuntimeOutcome
    summary: str
    evidence_refs: list[str]


@dataclass(slots=True)
class RuntimeMonitorRole:
    """First-class runtime monitoring role.

    This role is responsible for classifying run health into explicit categories
    (`healthy`, `soft_suspicion`, `waste_stop`, `hard_abort`) for downstream control
    decisions. Implementation remains intentionally stubbed in Stage 1.
    """

    name: str = "RuntimeMonitorRole"

    def run(self, run_id: str) -> RuntimeMonitorResult:
        """Classify current runtime state for the given run."""
        # TODO: implement monitoring logic and evidence collection.
        return RuntimeMonitorResult(
            outcome=RuntimeOutcome.SOFT_SUSPICION,
            summary="todo: runtime monitoring not implemented yet",
            evidence_refs=[],
        )
