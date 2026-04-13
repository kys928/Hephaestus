"""Schema: PreprocessingReport."""

from __future__ import annotations

from dataclasses import dataclass

from ._base import JsonSchema


@dataclass(slots=True)
class PreprocessingReport(JsonSchema):
    report_id: str
    manifest_id: str
    steps: list[str]
