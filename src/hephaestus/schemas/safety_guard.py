from __future__ import annotations

from dataclasses import dataclass, field

from ._base import JsonSchema


@dataclass(slots=True)
class SafetyGuardInput(JsonSchema):
    guard_id: str
    run_id: str
    lineage_id: str
    boundary: str
    payload: dict[str, object] = field(default_factory=dict)
    context: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class SafetyGuardResult(JsonSchema):
    guard_id: str
    run_id: str
    lineage_id: str
    boundary: str
    allowed: bool
    severity: str = "info"
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class SafetyPolicyDecision(JsonSchema):
    run_id: str
    lineage_id: str
    boundary: str
    allowed: bool
    effective_action: str | None = None
    blocking_guards: list[str] = field(default_factory=list)
    guard_results: list[dict[str, object]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
