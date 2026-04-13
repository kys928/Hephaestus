"""Backend contract definitions (backend-agnostic)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class BackendRunHandle:
    run_id: str
    session_id: str


class TrainingBackend(Protocol):
    """Generic training backend contract without provider-specific assumptions."""

    name: str

    def launch(self, launch_config: dict[str, object]) -> BackendRunHandle:
        ...

    def stop(self, run_id: str) -> None:
        ...


class EvaluationBackend(Protocol):
    """Generic evaluation backend contract."""

    name: str

    def run_eval(self, eval_config: dict[str, object]) -> dict[str, object]:
        ...
