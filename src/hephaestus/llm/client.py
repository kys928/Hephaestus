from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class LLMClient(Protocol):
    def complete(self, prompt: str, *, model: str, metadata: dict[str, object] | None = None) -> dict[str, object]: ...


@dataclass(slots=True)
class BoundaryOnlyLLMClient:
    """Deterministic boundary client; records request shape without external calls."""

    provider: str = "boundary_only"
    default_model: str = "offline-deterministic"
    calls: list[dict[str, object]] = field(default_factory=list)

    def complete(self, prompt: str, *, model: str | None = None, metadata: dict[str, object] | None = None) -> dict[str, object]:
        request = {"provider": self.provider, "model": model or self.default_model, "prompt_length": len(prompt), "metadata": dict(metadata or {})}
        self.calls.append(request)
        return {"text": "", "finish_reason": "boundary_only_no_external_call", "request": request}
