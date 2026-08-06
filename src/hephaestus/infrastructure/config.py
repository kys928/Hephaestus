"""Environment-driven infrastructure configuration without .env loading."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .secrets import SecretReference


class InfrastructureConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class InfrastructureConfig:
    state_root: Path
    artifact_root: Path
    event_log_path: Path
    worker_id: str
    lease_seconds: int = 60
    queue_backend: str = "memory"
    database_backend: str = "sqlite"
    sqlite_path: Path = Path("state/infrastructure.sqlite3")
    database_dsn_ref: SecretReference | None = None
    queue_poll_seconds: float = 1.0
    maximum_attempts: int = 3
    maximum_lease_expirations: int = 3
    artifact_backend: str = "filesystem"
    object_store_bucket: str | None = None
    object_store_prefix: str = "hephaestus"
    object_store_sse: str | None = "AES256"
    secrets_backend: str = "environment"
    secrets_root: Path = Path("/run/secrets")
    telemetry_backend: str = "jsonl"
    health_required: tuple[str, ...] = ("state_root", "artifact_root")
    shutdown_timeout_seconds: float = 30.0

    @staticmethod
    def _positive_int(values: Mapping[str, str], name: str, default: str) -> int:
        try:
            value = int(values.get(name, default))
        except ValueError as exc:
            raise InfrastructureConfigError(f"{name} must be an integer") from exc
        if value <= 0:
            raise InfrastructureConfigError(f"{name} must be positive")
        return value

    @staticmethod
    def _positive_float(values: Mapping[str, str], name: str, default: str) -> float:
        try:
            value = float(values.get(name, default))
        except ValueError as exc:
            raise InfrastructureConfigError(f"{name} must be numeric") from exc
        if value <= 0:
            raise InfrastructureConfigError(f"{name} must be positive")
        return value

    @staticmethod
    def _choice(values: Mapping[str, str], name: str, default: str, choices: set[str]) -> str:
        value = values.get(name, default).strip().lower()
        if value not in choices:
            raise InfrastructureConfigError(
                f"{name} must be one of: {', '.join(sorted(choices))}"
            )
        return value

    @staticmethod
    def _secret_reference(raw: str | None, name: str) -> SecretReference | None:
        if raw is None or not raw.strip():
            return None
        provider, separator, key = raw.partition(":")
        if not separator or not provider.strip() or not key.strip():
            raise InfrastructureConfigError(f"{name} must use provider:key reference syntax")
        return SecretReference(provider.strip(), key.strip())

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
        lease_seconds = cls._positive_int(
            values, "HEPHAESTUS_JOB_LEASE_SECONDS", "60"
        )
        queue_backend = cls._choice(
            values,
            "HEPHAESTUS_QUEUE_BACKEND",
            "memory",
            {"memory", "sqlite", "postgres"},
        )
        database_backend = cls._choice(
            values,
            "HEPHAESTUS_DATABASE_BACKEND",
            "sqlite",
            {"sqlite", "postgres"},
        )
        sqlite_path = Path(
            values.get(
                "HEPHAESTUS_SQLITE_PATH",
                str(state_root / "infrastructure.sqlite3"),
            )
        )
        database_dsn_ref = cls._secret_reference(
            values.get("HEPHAESTUS_DATABASE_DSN_REF"),
            "HEPHAESTUS_DATABASE_DSN_REF",
        )
        queue_poll_seconds = cls._positive_float(
            values, "HEPHAESTUS_QUEUE_POLL_SECONDS", "1"
        )
        maximum_attempts = cls._positive_int(
            values, "HEPHAESTUS_JOB_MAXIMUM_ATTEMPTS", "3"
        )
        maximum_lease_expirations = cls._positive_int(
            values, "HEPHAESTUS_MAXIMUM_LEASE_EXPIRATIONS", "3"
        )
        artifact_backend = cls._choice(
            values,
            "HEPHAESTUS_ARTIFACT_BACKEND",
            "filesystem",
            {"filesystem", "s3"},
        )
        object_store_bucket = values.get("HEPHAESTUS_OBJECT_STORE_BUCKET") or None
        object_store_prefix = values.get(
            "HEPHAESTUS_OBJECT_STORE_PREFIX", "hephaestus"
        ).strip("/")
        object_store_sse = values.get("HEPHAESTUS_OBJECT_STORE_SSE", "AES256") or None
        secrets_backend = cls._choice(
            values,
            "HEPHAESTUS_SECRETS_BACKEND",
            "environment",
            {"environment", "file", "aws-secrets-manager", "injected"},
        )
        secrets_root = Path(values.get("HEPHAESTUS_SECRETS_ROOT", "/run/secrets"))
        telemetry_backend = cls._choice(
            values,
            "HEPHAESTUS_TELEMETRY_BACKEND",
            "jsonl",
            {"jsonl", "none", "opentelemetry"},
        )
        health_required = tuple(
            part.strip()
            for part in values.get(
                "HEPHAESTUS_HEALTH_REQUIRED", "state_root,artifact_root"
            ).split(",")
            if part.strip()
        )
        allowed_health = {
            "state_root",
            "artifact_root",
            "database",
            "queue",
            "artifact_store",
            "lock_service",
            "secret_provider",
            "worker",
            "migrations",
        }
        unknown = set(health_required) - allowed_health
        if unknown:
            raise InfrastructureConfigError(
                f"HEPHAESTUS_HEALTH_REQUIRED contains unknown checks: {sorted(unknown)}"
            )
        shutdown_timeout_seconds = cls._positive_float(
            values, "HEPHAESTUS_SHUTDOWN_TIMEOUT_SECONDS", "30"
        )
        if queue_backend == "postgres" and database_dsn_ref is None:
            raise InfrastructureConfigError(
                "PostgreSQL queue configuration requires HEPHAESTUS_DATABASE_DSN_REF"
            )
        if artifact_backend == "s3" and not object_store_bucket:
            raise InfrastructureConfigError(
                "S3 artifact configuration requires HEPHAESTUS_OBJECT_STORE_BUCKET"
            )
        return cls(
            state_root,
            artifact_root,
            event_log_path,
            worker_id,
            lease_seconds,
            queue_backend,
            database_backend,
            sqlite_path,
            database_dsn_ref,
            queue_poll_seconds,
            maximum_attempts,
            maximum_lease_expirations,
            artifact_backend,
            object_store_bucket,
            object_store_prefix,
            object_store_sse,
            secrets_backend,
            secrets_root,
            telemetry_backend,
            health_required,
            shutdown_timeout_seconds,
        )
