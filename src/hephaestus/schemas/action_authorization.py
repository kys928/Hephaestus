"""Scope of an operator approval, bound to an exact action and state."""
from dataclasses import dataclass
from ._base import JsonSchema


@dataclass(slots=True)
class ActionAuthorizationScope(JsonSchema):
    action: str
    lineage_id: str
    run_id: str
    stage_name: str
    checkpoint_ref: str | None
    lineage_sha256: str
    expires_at: str
    child_lineage_id: str | None = None
    eval_report_ref: str | None = None
    data_manifest_ref: str | None = None
