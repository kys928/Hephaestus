from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from ._base import JsonSchema


@dataclass(slots=True)
class LineageState(JsonSchema):
    """Compact current lineage truth for multi-run control.

    Historical run/decision details remain append-only in run/decision stores.
    """

    lineage_id: str
    parent_lineage_id: str | None = None
    child_lineage_ids: list[str] = field(default_factory=list)
    stage_name: str = ""
    status: str = "exploratory"
    trust_level: str = "unknown"
    origin_run_id: str | None = None
    origin_checkpoint_ref: str | None = None
    branch_origin_checkpoint_ref: str | None = None
    created_at: str | None = None
    updated_at: str = ""
    architecture_contract_ref: str | None = None
    tokenizer_contract_ref: str | None = None
    data_policy_ref: str | None = None
    training_recipe_ref: str | None = None
    eval_policy_ref: str | None = None
    loop_index: int = 0
    run_count: int = 0
    latest_run_id: str | None = None
    best_checkpoint_ref: str | None = None
    last_stable_checkpoint_ref: str | None = None
    certified_stable_checkpoint_ref: str | None = None
    last_certification_result: str | None = "certification_not_eligible"
    last_decision: str | None = None
    last_decision_id: str | None = None
    last_requested_action: str | None = None
    last_effective_action: str | None = None
    last_approval_status: str | None = "none"
    pending_approval: bool = False
    last_high_impact_request_id: str | None = None
    last_repeated_eval_count: int = 0
    last_consistency_score: float = 0.0
    last_variance_risk: str = "unknown"
    certification_recheck_count: int = 0
    repeatability_sufficient: bool = False
    recent_failures: list[str] = field(default_factory=list)
    known_pathologies: list[str] = field(default_factory=list)
    major_interventions: list[dict[str, object]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LineageState":
        keys = {field_.name for field_ in fields(cls)}
        filtered = {key: value for key, value in payload.items() if key in keys}
        return cls(**filtered)
