from __future__ import annotations

from dataclasses import dataclass, field

from ._base import JsonSchema


@dataclass(slots=True)
class PreprocessingReport(JsonSchema):
    report_id: str
    run_id: str
    manifest_id: str
    operations: list[str] = field(default_factory=list)
    processed_dataset_ref: str = ""
    dropped_examples: int = 0
