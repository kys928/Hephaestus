from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._base import JsonSchema


@dataclass(slots=True)
class SafetyGuardResult(JsonSchema):
    guard_id: str
    passed: bool
    severity: str = "info"
    reasons: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SafetyGuardResult":
        return cls(
            guard_id=str(payload.get("guard_id") or "guard"),
            passed=bool(payload.get("passed", False)),
            severity=str(payload.get("severity") or "info"),
            reasons=[str(v) for v in payload.get("reasons", [])] if isinstance(payload.get("reasons"), list) else [],
            evidence_refs=[str(v) for v in payload.get("evidence_refs", [])] if isinstance(payload.get("evidence_refs"), list) else [],
            metadata=dict(payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}),
        )
