"""Schema: CheckpointResolution."""

from __future__ import annotations

from dataclasses import dataclass

from ._base import JsonSchema


@dataclass(slots=True)
class CheckpointResolution(JsonSchema):
    run_id: str
    checkpoint_path: str
    compatible: bool
