from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PromotionPolicy:
    min_confidence_for_best: float = 0.6
    min_confidence_for_stable: float = 0.85

    def decide(self, deterministic_passed: bool, confidence: float, has_candidate: bool = True) -> str:
        if not has_candidate:
            return "inconclusive"
        if not deterministic_passed:
            return "rejected"
        if confidence >= self.min_confidence_for_stable:
            return "stable"
        if confidence >= self.min_confidence_for_best:
            return "promoted_best"
        return "candidate_best"
