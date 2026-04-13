"""Schema: BackendContract."""

from __future__ import annotations

from dataclasses import dataclass

from ._base import JsonSchema


@dataclass(slots=True)
class BackendContract(JsonSchema):
    backend_name: str
    supports_training: bool
    supports_eval: bool
