from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ._base import JsonSchema


ReplayVerificationStatus = Literal["reproducible", "partial", "insufficient", "missing"]


@dataclass(slots=True)
class ReplayVerificationReport(JsonSchema):
    run_id: str
    lineage_id: str | None
    status: ReplayVerificationStatus
    checked_at: str
    evidence_refs: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    replay_scope: str = "unknown"
    checkpoint_ref: str | None = None
    checkpoint_content_hash: str | None = None
    content_hash_available: bool = False
    requires_content_hash_match: bool = False
    manifest_id: str | None = None
    eval_report_id: str | None = None
    decision_id: str | None = None
    confidence_ceiling: float | None = None
    summary: str = ""
