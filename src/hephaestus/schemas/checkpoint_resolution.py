from __future__ import annotations

from dataclasses import dataclass

from ._base import JsonSchema


@dataclass(slots=True)
class CheckpointResolution(JsonSchema):
    run_id: str
    selected_checkpoint_ref: str
    reason: str
    confidence: float
