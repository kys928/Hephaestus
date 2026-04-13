from __future__ import annotations

from dataclasses import dataclass, field

from ._base import JsonSchema


@dataclass(slots=True)
class BackendContract(JsonSchema):
    backend_name: str
    dry_run: bool
    supports_training: bool
    supports_runtime_events: bool
    supports_eval_artifacts: bool
    expected_artifact_kinds: list[str] = field(default_factory=list)
