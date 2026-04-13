"""Backend contract definitions (backend-agnostic)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from hephaestus.schemas.runtime_event import RuntimeEvent


@dataclass(slots=True)
class BackendTarget:
    backend_name: str
    dry_run: bool
    config: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class PreparedBackendJob:
    run_id: str
    backend_name: str
    artifact_root: str
    expected_artifacts: list[str] = field(default_factory=list)
    execution_spec: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class BackendRunResult:
    run_id: str
    status: str
    events: list[RuntimeEvent] = field(default_factory=list)
    artifact_refs: list[str] = field(default_factory=list)
    checkpoint_candidates: list[dict[str, object]] = field(default_factory=list)
    intermediate_eval: dict[str, object] = field(default_factory=dict)


class ExecutionBackend(Protocol):
    """Generic backend contract that supports bounded execution responsibilities."""

    name: str

    def resolve_target(self, launch_config: dict[str, object]) -> BackendTarget:
        ...

    def acquire_dataset(self, run_id: str) -> dict[str, object]:
        ...

    def preprocess(self, run_id: str) -> dict[str, object]:
        ...

    def prepare_training_job(
        self,
        *,
        experiment_plan: dict[str, object],
        data_contract: dict[str, object],
        training_plan: dict[str, object],
        launch_config: dict[str, object],
    ) -> PreparedBackendJob:
        ...

    def launch_training(self, prepared_job: PreparedBackendJob) -> BackendRunResult:
        ...

    def stop(self, run_id: str) -> None:
        ...
