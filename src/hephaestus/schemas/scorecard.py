from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from ._base import JsonSchema


@dataclass(slots=True)
class Scorecard(JsonSchema):
    scorecard_id: str
    run_id: str
    eval_pack_id: str | None = None
    eval_pack_version: str | None = None
    checkpoint_ref: str | None = None
    created_at: str | None = None
    deterministic_passed: bool = False
    failed_gates: list[str] = field(default_factory=list)
    passed_gates: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    gate_results: dict[str, dict[str, object]] = field(default_factory=dict)
    repetition_passed: bool | None = None
    length_termination_passed: bool | None = None
    structure_passed: bool | None = None
    continuation_passed: bool | None = None
    ranking_passed: bool | None = None
    evidence_refs: list[str] = field(default_factory=list)
    scorecard_integrity_level: str = "insufficient"
    completeness_score: float = 0.0
    missing_fields: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Scorecard":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"scorecard payload has unknown fields: {', '.join(unknown)}")
        return cls(**payload)

    def enforce_semantics(self) -> "Scorecard":
        self.failed_gates = [str(item) for item in self.failed_gates]
        self.passed_gates = [str(item) for item in self.passed_gates]
        if self.failed_gates:
            self.deterministic_passed = False
        expected = ["deterministic_passed", "gate_results", "metrics", "thresholds"]
        missing = [field for field in expected if not getattr(self, field)]
        self.missing_fields = sorted(set(self.missing_fields + missing))

        completeness = 1.0 - (0.2 * len(self.missing_fields))
        if self.ranking_passed is None:
            completeness -= 0.05
        if self.continuation_passed is None:
            completeness -= 0.05
        self.completeness_score = max(0.0, min(1.0, completeness))

        if not self.eval_pack_id:
            self.scorecard_integrity_level = "incomplete_eval_pack_identity"
            self.warnings.append("eval_pack_id_missing")
        elif self.scorecard_integrity_level not in {"content_hash_verified", "reference_only", "inline_unhashed"}:
            self.scorecard_integrity_level = "inline_unhashed"

        if not self.gate_results:
            self.warnings.append("deterministic_gate_results_missing")

        return self
