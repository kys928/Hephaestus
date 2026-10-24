"""Explicit binding between a campaign run and immutable source evidence."""
from dataclasses import dataclass
from ._base import JsonSchema


@dataclass(slots=True)
class ScientificEvidenceBinding(JsonSchema):
    binding_id: str
    target_run_id: str
    source_run_id: str
    lineage_id: str
    stage_name: str
    kind: str
    artifact_ref: str
    superseded_by: str | None = None
