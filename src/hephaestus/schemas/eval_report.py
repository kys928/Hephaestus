"""Schema: EvalReport."""

from __future__ import annotations

from dataclasses import dataclass

from ._base import JsonSchema


@dataclass(slots=True)
class EvalReport(JsonSchema):
    eval_id: str
    run_id: str
    pack_name: str
    artifact_path: str
