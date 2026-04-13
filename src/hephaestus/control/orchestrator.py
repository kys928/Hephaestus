"""Orchestrator skeleton that wires explicit role order."""

from __future__ import annotations

from dataclasses import dataclass

from hephaestus.control.spine import SPINE_ORDER, PhaseResult, SpineCoordinator, SpinePhase


@dataclass(slots=True)
class Orchestrator:
    """Minimal orchestrator: enforces ordering, delegates phase execution."""

    coordinator: SpineCoordinator

    def run(self, run_id: str) -> list[PhaseResult]:
        results: list[PhaseResult] = []
        for phase in SPINE_ORDER:
            result = self.coordinator.run_phase(phase=phase, run_id=run_id)
            results.append(result)
            if phase is SpinePhase.JUDGE_EXIT:
                break
        return results
