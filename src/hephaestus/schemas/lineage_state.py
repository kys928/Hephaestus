from __future__ import annotations

from dataclasses import dataclass, field

from ._base import JsonSchema


@dataclass(slots=True)
class LineageState(JsonSchema):
    """Compact current lineage truth for multi-run control.

    Historical run/decision details remain append-only in run/decision stores.
    """

    lineage_id: str
    parent_lineage_id: str | None
    stage_name: str
    status: str
    trust_level: str = "unknown"
    loop_index: int = 0
    latest_run_id: str | None = None
    best_checkpoint_ref: str | None = None
    last_stable_checkpoint_ref: str | None = None
    certified_stable_checkpoint_ref: str | None = None
    last_certification_result: str = "certification_not_eligible"
    last_repeated_eval_count: int = 0
    last_consistency_score: float = 0.0
    last_variance_risk: str = "unknown"
    certification_recheck_count: int = 0
    repeatability_sufficient: bool = False
    recent_failures: list[str] = field(default_factory=list)
    known_pathologies: list[str] = field(default_factory=list)
    last_decision: str | None = None
    last_decision_id: str | None = None
    branch_origin_checkpoint_ref: str | None = None
    child_lineage_ids: list[str] = field(default_factory=list)
    run_count: int = 0
    updated_at: str = ""
