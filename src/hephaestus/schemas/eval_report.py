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
    intermediate_artifact_refs: list[str] = field(default_factory=list)
