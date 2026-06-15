from __future__ import annotations

from dataclasses import dataclass, field

from ._base import JsonSchema


@dataclass(slots=True)
class OperatorConsolePayload(JsonSchema):
    read_only: bool = True
    status: str = "ok"
    runs: list[dict[str, object]] = field(default_factory=list)
    run: dict[str, object] = field(default_factory=dict)
    error: str | None = None
