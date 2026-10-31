from __future__ import annotations

from dataclasses import dataclass
from ._base import JsonSchema


@dataclass(slots=True)
class DatasetRecordMapping(JsonSchema):
    prompt_fields: list[str]
    target_field: str
    raw_text: bool = False
