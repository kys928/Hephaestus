from __future__ import annotations

from dataclasses import dataclass, field

from ._base import JsonSchema


@dataclass(slots=True)
class EvalReport(JsonSchema):
    eval_id: str
    run_id: str
    stage_name: str
    pack_name: str
    metric_summaries: list[dict[str, object]] = field(default_factory=list)
    regression_summary: dict[str, object] = field(default_factory=dict)
    checkpoint_resolution: dict[str, object] = field(default_factory=dict)
    confidence: float = 0.0
    likely_issue_category: str = "none"
    evidence_completeness: float = 0.0
    stability_confidence: float = 0.0
    certification_readiness: str = "certification_not_eligible"
    recheck_recommended: bool = False
    promotion_bundle_passed: bool = False
    observed_consistent_runs: int = 0
    repeated_eval_count: int = 0
    consistency_score: float = 0.0
    repeatability_ready: bool = False
    repeatability_blocked: bool = False
    repeatability_sufficient: bool = False
    recheck_needed: bool = False
    variance_risk: str = "unknown"
    consistency_observed: str = "unknown"
    certification_recheck_count: int = 0
    evaluation_bundle_summary: dict[str, object] = field(default_factory=dict)
    intermediate_artifact_refs: list[str] = field(default_factory=list)
