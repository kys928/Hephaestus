from __future__ import annotations

from dataclasses import dataclass, field

from ._base import JsonSchema


@dataclass(slots=True)
class StageProfile(JsonSchema):
    name: str
    strictness: str
    eval_pack: str
    deterministic_gates: dict[str, float]
    allowed_next_actions: list[str] = field(default_factory=list)
