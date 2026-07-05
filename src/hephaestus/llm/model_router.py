from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ModelRoute:
    task: str
    model: str
    provider: str = "boundary_only"
    requires_approval: bool = False


@dataclass(slots=True)
class ModelRouter:
    routes: dict[str, ModelRoute] = field(default_factory=dict)
    fallback_model: str = "offline-deterministic"

    def resolve(self, task: str) -> ModelRoute:
        return self.routes.get(task, ModelRoute(task=task, model=self.fallback_model))
