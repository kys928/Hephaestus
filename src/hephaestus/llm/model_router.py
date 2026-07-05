"""Model routing for role-scoped LLM calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from hephaestus.config_loader import ConfigError, load_config_file


class ModelRouterError(ConfigError):
    """Raised when no deterministic LLM route can be resolved."""


@dataclass(frozen=True, slots=True)
class LLMRoute:
    role: str
    model: str
    provider: str = "manual"
    parameters: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {"role": self.role, "model": self.model, "provider": self.provider, "parameters": dict(self.parameters)}


@dataclass(frozen=True, slots=True)
class ModelRouter:
    """Resolve role names to concrete model/provider routes from JSON-compatible config."""

    routes: Mapping[str, LLMRoute]
    default_route: LLMRoute | None = None

    @classmethod
    def from_config(cls, config_dir: Path = Path("configs"), config_name: str = "llm") -> "ModelRouter":
        path = config_dir / f"{config_name}.yaml"
        if not path.exists():
            return cls(routes={}, default_route=LLMRoute(role="default", model="manual-review", provider="manual"))
        payload = load_config_file(path)
        default_route = _route_from_payload("default", payload.get("default", {})) if isinstance(payload.get("default", {}), dict) else None
        route_payloads = payload.get("routes", {})
        if not isinstance(route_payloads, dict):
            raise ModelRouterError("LLM route config field 'routes' must be an object")
        routes = {str(role): _route_from_payload(str(role), route) for role, route in route_payloads.items() if isinstance(route, dict)}
        return cls(routes=routes, default_route=default_route)

    def resolve(self, role: str, overrides: Mapping[str, object] | None = None) -> LLMRoute:
        base = self.routes.get(role) or self.default_route
        if base is None:
            raise ModelRouterError(f"no LLM route configured for role '{role}' and no default route")
        if not overrides:
            return base
        params = dict(base.parameters)
        params.update(dict(overrides.get("parameters", {})) if isinstance(overrides.get("parameters", {}), dict) else {})
        return LLMRoute(
            role=role,
            model=str(overrides.get("model", base.model)),
            provider=str(overrides.get("provider", base.provider)),
            parameters=params,
        )


def _route_from_payload(role: str, payload: Mapping[str, object]) -> LLMRoute:
    model = str(payload.get("model", "")).strip()
    if not model:
        raise ModelRouterError(f"LLM route '{role}' requires a non-empty model")
    provider = str(payload.get("provider", "manual")).strip() or "manual"
    parameters = payload.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ModelRouterError(f"LLM route '{role}' parameters must be an object")
    return LLMRoute(role=role, model=model, provider=provider, parameters=dict(parameters))
