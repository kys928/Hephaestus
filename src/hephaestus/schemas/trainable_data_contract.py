"""Schema: TrainableDataContract."""

from __future__ import annotations

from dataclasses import dataclass

from ._base import JsonSchema


@dataclass(slots=True)
class TrainableDataContract(JsonSchema):
    contract_id: str
    manifest_id: str
    schema_version: str
