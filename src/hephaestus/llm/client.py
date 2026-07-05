"""Typed LLM client boundary for role-scoped structured calls."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, TypeVar

from hephaestus.llm.model_router import LLMRoute, ModelRouter
from hephaestus.llm.prompt_loader import PromptLoader
from hephaestus.llm.retry import RetryPolicy, run_with_retry
from hephaestus.llm.structured_output import validate_structured_output
from hephaestus.schemas._base import JsonSchema

T = TypeVar("T", bound=JsonSchema)


class LLMTransport(Protocol):
    def complete(self, *, route: LLMRoute, prompt: str, schema_name: str) -> str | dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class LLMRequest:
    role: str
    prompt_name: str
    schema_name: str
    context: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {"role": self.role, "prompt_name": self.prompt_name, "schema_name": self.schema_name, "context": dict(self.context)}


@dataclass(frozen=True, slots=True)
class LLMResponse:
    request: dict[str, object]
    route: dict[str, object]
    output: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {"request": dict(self.request), "route": dict(self.route), "output": dict(self.output)}


class MissingLLMTransportError(RuntimeError):
    """Raised when a routed call has no concrete transport implementation."""


@dataclass(slots=True)
class LLMClient:
    """Load prompts, route model calls, retry transport failures, and validate schemas."""

    prompt_loader: PromptLoader = field(default_factory=PromptLoader)
    router: ModelRouter = field(default_factory=ModelRouter.from_config)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    transport: LLMTransport | Callable[..., str | dict[str, Any]] | None = None

    @classmethod
    def from_paths(cls, prompt_dir: Path = Path("prompts"), config_dir: Path = Path("configs")) -> "LLMClient":
        return cls(prompt_loader=PromptLoader(prompt_dir), router=ModelRouter.from_config(config_dir))

    def generate_structured(
        self,
        *,
        role: str,
        prompt_name: str,
        schema: type[T],
        context: Mapping[str, object] | None = None,
        route_overrides: Mapping[str, object] | None = None,
    ) -> T:
        request = LLMRequest(role=role, prompt_name=prompt_name, schema_name=schema.__name__, context=dict(context or {}))
        prompt = self.prompt_loader.render(prompt_name, {"context_json": json.dumps(request.context, sort_keys=True), **request.context})
        route = self.router.resolve(role, route_overrides)

        def _call() -> str | dict[str, Any]:
            if self.transport is None:
                raise MissingLLMTransportError("LLMClient requires an explicit transport for non-dry-run calls")
            if hasattr(self.transport, "complete"):
                return self.transport.complete(route=route, prompt=prompt, schema_name=schema.__name__)  # type: ignore[union-attr]
            return self.transport(route=route, prompt=prompt, schema_name=schema.__name__)  # type: ignore[misc]

        raw = run_with_retry(_call, self.retry_policy)
        return validate_structured_output(raw, schema)  # type: ignore[return-value]

    def generate_structured_response(self, **kwargs: Any) -> LLMResponse:
        result = self.generate_structured(**kwargs)
        route = self.router.resolve(str(kwargs["role"]), kwargs.get("route_overrides"))
        request = LLMRequest(
            role=str(kwargs["role"]),
            prompt_name=str(kwargs["prompt_name"]),
            schema_name=kwargs["schema"].__name__,
            context=dict(kwargs.get("context") or {}),
        )
        return LLMResponse(request=request.to_dict(), route=route.to_dict(), output=result.to_dict())
