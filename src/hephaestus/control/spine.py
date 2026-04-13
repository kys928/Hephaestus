"""Mandatory control spine definitions and coordinator interface."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class SpinePhase(str, Enum):
    JUDGE_ENTRY = "judge_entry"
    PLANNER = "planner"
    DATA_ACQUISITION_AUDIT = "data_acquisition_audit"
    DATA_PREPROCESSOR = "data_preprocessor"
    TRAINING_ENGINEER = "training_engineer"
    RUNTIME_MONITOR = "runtime_monitor"
    EVALUATOR = "evaluator"
    JUDGE_EXIT = "judge_exit"


SPINE_ORDER: tuple[SpinePhase, ...] = (
    SpinePhase.JUDGE_ENTRY,
    SpinePhase.PLANNER,
    SpinePhase.DATA_ACQUISITION_AUDIT,
    SpinePhase.DATA_PREPROCESSOR,
    SpinePhase.TRAINING_ENGINEER,
    SpinePhase.RUNTIME_MONITOR,
    SpinePhase.EVALUATOR,
    SpinePhase.JUDGE_EXIT,
)


@dataclass(slots=True)
class PhaseResult:
    phase: SpinePhase
    status: str
    artifact_refs: list[str]
    output: dict[str, object] | None = None


class SpineCoordinator(Protocol):
    def run_phase(self, phase: SpinePhase, run_id: str) -> PhaseResult: ...
