from __future__ import annotations

from dataclasses import dataclass

from ._base import JsonSchema


@dataclass(slots=True)
class TrainableDataContract(JsonSchema):
    contract_id: str
    run_id: str
    manifest_id: str
    processed_dataset_ref: str
    schema_version: str
    min_tokens: int
