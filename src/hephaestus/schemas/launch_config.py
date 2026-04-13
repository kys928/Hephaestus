from __future__ import annotations

from dataclasses import dataclass

from ._base import JsonSchema


@dataclass(slots=True)
class LaunchConfig(JsonSchema):
    launch_id: str
    run_id: str
    backend: str
    dry_run: bool
    artifact_root: str
    parameters: dict[str, str]
