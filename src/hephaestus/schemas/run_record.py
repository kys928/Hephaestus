"""Schema: RunRecord."""

from __future__ import annotations

from dataclasses import dataclass

from ._base import JsonSchema


@dataclass(slots=True)
class RunRecord(JsonSchema):
    run_id: str
    target_id: str
    stage: str
    status: str
    artifact_root: str
