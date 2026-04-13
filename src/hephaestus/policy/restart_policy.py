"""Policy module: RestartPolicy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RestartPolicy:
    """Policy stub with explicit, reviewable decisions."""

    def decide(self, context: dict[str, object]) -> dict[str, object]:
        # TODO: replace with deterministic policy evaluation.
        return {"policy": "RestartPolicy", "decision": "todo", "context": context}
