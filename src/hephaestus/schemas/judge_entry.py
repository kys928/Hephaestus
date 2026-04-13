"""Schema: JudgeEntry."""

from __future__ import annotations

from dataclasses import dataclass

from ._base import JsonSchema


@dataclass(slots=True)
class JudgeEntry(JsonSchema):
    run_id: str
    objective: str
    constraints: list[str]
