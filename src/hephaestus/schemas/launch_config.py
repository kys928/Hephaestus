"""Schema: LaunchConfig."""

from __future__ import annotations

from dataclasses import dataclass

from ._base import JsonSchema


@dataclass(slots=True)
class LaunchConfig(JsonSchema):
    launch_id: str
    backend: str
    parameters: dict[str, str]
