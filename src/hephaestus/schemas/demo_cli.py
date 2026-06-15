from __future__ import annotations

from dataclasses import dataclass, field

from ._base import JsonSchema


@dataclass(slots=True)
class DoctorPayload(JsonSchema):
    state_root: str
    exists: bool
    is_dir: bool
    created: bool = False
    status: str = "missing"
    checks: list[dict[str, object]] = field(default_factory=list)


@dataclass(slots=True)
class DemoStatePayload(JsonSchema):
    state_root: str
    run_id: str
    lineage_id: str
    created: bool = True
