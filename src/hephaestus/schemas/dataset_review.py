"""Explicit evidence and access scopes; neither record grants its own trust."""
from __future__ import annotations

from dataclasses import dataclass, field

from ._base import JsonSchema


@dataclass(slots=True)
class DatasetSourceReview(JsonSchema):
    candidate_sha256: str
    reviewer_id: str
    scope: str  # public_preview or full_corpus
    revision_verified: bool
    schema_status: str
    language_status: str
    capability_status: str
    rights_status: str  # unresolved, declared_only, cleared
    rights_evidence_refs: list[str]
    tokenizer_ref: str
    tokenizer_file_sha256: str
    total_tokens: int
    unknown_tokens: int
    normalization_profile: str
    source_file_hashes: dict[str, str] = field(default_factory=dict)
    audit_refs: dict[str, str] = field(default_factory=dict)
    reference_manifest_refs: dict[str, str] = field(default_factory=dict)
    access_scope_ref: str | None = None
    processing_profile_sha256: str | None = None
    limitations: list[str] = field(default_factory=list)
    schema: str = "dataset-source-review.v1"


@dataclass(slots=True)
class DatasetScreeningAccessScope(JsonSchema):
    candidate_sha256: str
    acquisition_files: list[str]
    maximum_bytes: int
    purposes: list[str]
    reference_manifest_refs: dict[str, str]
    status: str = "requested"
    authorized_by: str | None = None
    training_allowed: bool = False
    schema: str = "dataset-screening-access-scope.v1"
