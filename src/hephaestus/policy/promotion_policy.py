from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PromotionPolicy:
    def decide(self, deterministic_passed: bool, confidence: float) -> str:
        if not deterministic_passed:
            return "reject_checkpoint"
        return "promote_checkpoint" if confidence >= 0.6 else "continue_from_checkpoint"
