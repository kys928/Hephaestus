from __future__ import annotations

from dataclasses import dataclass, field

from ._base import JsonSchema


@dataclass(slots=True)
class StageProfile(JsonSchema):
    name: str
    strictness: str
    eval_pack: str
    deterministic_gates: dict[str, float]
    allowed_next_actions: list[str] = field(default_factory=list)
    certification_profile: dict[str, object] = field(default_factory=dict)
    eval_pack_ref: str | None = None
    required_evidence: dict[str, int] = field(default_factory=dict)
    stage_thresholds: dict[str, float] = field(default_factory=dict)
    deterministic_gate_config: dict[str, object] = field(default_factory=dict)
