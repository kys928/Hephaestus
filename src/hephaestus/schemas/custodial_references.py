"""Frozen inventory metadata. No protected prompts, labels, or private keys."""
from dataclasses import dataclass
from ._base import JsonSchema


@dataclass(slots=True)
class CustodialReferenceManifest(JsonSchema):
    role: str
    custodian_id: str
    content_ref: str
    row_count: int
    sealed_at: str
    normalization_profile: str
    original_text_available_to_custodian: bool
    exposure_inventory_complete: bool
    schema: str = "custodial-reference-manifest.v1"
