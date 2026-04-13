"""Schema: StageProfile."""

from __future__ import annotations

from dataclasses import dataclass

from ._base import JsonSchema


@dataclass(slots=True)
class StageProfile(JsonSchema):
    name: str
    allowed_transitions: list[str]
    eval_pack: str
