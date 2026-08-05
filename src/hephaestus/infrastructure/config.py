"""Environment-driven infrastructure configuration without .env loading."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class InfrastructureConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class InfrastructureConfig:
    state_root: Path
    artifact_root: Path
    event_log_path: Path
    worker_id: str
    lease_seconds: int = 60

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "InfrastructureConfig":
        values = environ if environ is not None else os.environ
        state_root = Path(values.get("HEPHAESTUS_STATE_ROOT", "state"))
        artifact_root = Path(values.get("HEPHAESTUS_ARTIFACT_ROOT", "artifacts"))
        event_log_path = Path(
            values.get("HEPHAESTUS_EVENT_LOG", str(state_root / "infrastructure_events.jsonl"))
        )
        worker_id = values.get("HEPHAESTUS_WORKER_ID", socket.gethostname()).strip()
        if not worker_id:
            raise InfrastructureConfigError("HEPHAESTUS_WORKER_ID must not be empty")
        try:
            lease_seconds = int(values.get("HEPHAESTUS_JOB_LEASE_SECONDS", "60"))
        except ValueError as exc:
            raise InfrastructureConfigError("HEPHAESTUS_JOB_LEASE_SECONDS must be an integer") from exc
        if lease_seconds <= 0:
            raise InfrastructureConfigError("HEPHAESTUS_JOB_LEASE_SECONDS must be positive")
        return cls(state_root, artifact_root, event_log_path, worker_id, lease_seconds)
