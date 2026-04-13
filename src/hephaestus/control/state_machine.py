from __future__ import annotations

from dataclasses import dataclass

from hephaestus.control.spine import SpinePhase
from hephaestus.control.transition_rules import next_phase


@dataclass(slots=True)
class StateMachine:
    current: SpinePhase = SpinePhase.JUDGE_ENTRY

    def advance(self) -> SpinePhase | None:
        nxt = next_phase(self.current)
        if nxt is None:
            return None
        self.current = nxt
        return nxt
