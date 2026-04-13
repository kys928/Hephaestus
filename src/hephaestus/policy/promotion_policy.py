"""Policy module: PromotionPolicy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PromotionPolicy:
    """Policy stub with explicit, reviewable decisions."""

    def decide(self, context: dict[str, object]) -> dict[str, object]:
        # TODO: replace with deterministic policy evaluation.
        return {"policy": "PromotionPolicy", "decision": "todo", "context": context}
