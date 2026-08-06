from __future__ import annotations

import json

import pytest

from hephaestus.infrastructure.cli import main
from hephaestus.infrastructure.config import InfrastructureConfig, InfrastructureConfigError


def _durable_environment(tmp_path) -> dict[str, str]:
    state = tmp_path / "state"
    artifacts = tmp_path / "artifacts"
    return {
        "HEPHAESTUS_STATE_ROOT": str(state),
        "HEPHAESTUS_ARTIFACT_ROOT": str(artifacts),
        "HEPHAESTUS_EVENT_LOG": str(state / "events.jsonl"),
        "HEPHAESTUS_WORKER_ID": "worker-1",
        "HEPHAESTUS_QUEUE_BACKEND": "sqlite",
        "HEPHAESTUS_DATABASE_BACKEND": "sqlite",
        "HEPHAESTUS_SQLITE_PATH": str(state / "infrastructure.sqlite3"),
        "HEPHAESTUS_HEALTH_REQUIRED": (
            "state_root,artifact_root,database,queue,artifact_store,"
            "lock_service,secret_provider,worker,migrations"
        ),
    }


def test_migration_smoke_then_component_readiness(monkeypatch, tmp_path, capsys) -> None:
    environment = _durable_environment(tmp_path)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    assert main(["health", "--json"]) == 1
    before = json.loads(capsys.readouterr().out)
    assert before["ready"] is False
    assert before["checks"]["migrations"] != "ok"

    assert main(["migrate", "--json"]) == 0
    migration = json.loads(capsys.readouterr().out)
    assert migration["migrated"] is True
    assert migration["versions"] == {"locks": 1, "queue": 1, "state": 1}

    assert main(["health", "--json"]) == 0
    after = json.loads(capsys.readouterr().out)
    assert after["ready"] is True
    for component in (
        "database",
        "queue",
        "artifact_store",
        "lock_service",
        "secret_provider",
        "worker",
        "migrations",
    ):
        assert after["checks"][component] == "ok"


def test_configuration_carries_references_not_raw_database_credentials(tmp_path) -> None:
    environment = _durable_environment(tmp_path)
    environment.update(
        {
            "HEPHAESTUS_QUEUE_BACKEND": "postgres",
            "HEPHAESTUS_DATABASE_BACKEND": "postgres",
            "HEPHAESTUS_DATABASE_DSN_REF": "file:postgres_dsn",
            "HEPHAESTUS_JOB_MAXIMUM_ATTEMPTS": "5",
            "HEPHAESTUS_MAXIMUM_LEASE_EXPIRATIONS": "4",
            "HEPHAESTUS_SHUTDOWN_TIMEOUT_SECONDS": "12.5",
        }
    )
    config = InfrastructureConfig.from_env(environment)

    assert config.database_dsn_ref.to_dict() == {
        "provider": "file",
        "key": "postgres_dsn",
    }
    assert config.maximum_attempts == 5
    assert config.maximum_lease_expirations == 4
    assert config.shutdown_timeout_seconds == 12.5
    assert "postgresql://" not in repr(config)


def test_configuration_rejects_missing_references_and_unknown_components(tmp_path) -> None:
    environment = _durable_environment(tmp_path)
    environment["HEPHAESTUS_QUEUE_BACKEND"] = "postgres"
    with pytest.raises(InfrastructureConfigError, match="DSN_REF"):
        InfrastructureConfig.from_env(environment)

    environment = _durable_environment(tmp_path)
    environment["HEPHAESTUS_HEALTH_REQUIRED"] = "queue,imaginary-service"
    with pytest.raises(InfrastructureConfigError, match="unknown checks"):
        InfrastructureConfig.from_env(environment)


def test_optional_adapter_modules_import_without_cloud_sdks() -> None:
    from hephaestus.infrastructure.observability import OpenTelemetryEventSink
    from hephaestus.infrastructure.secrets import AwsSecretsManagerProvider
    from hephaestus.storage import PostgresLeaseLockProvider, S3ArtifactStore

    assert OpenTelemetryEventSink.__name__ == "OpenTelemetryEventSink"
    assert AwsSecretsManagerProvider.__name__ == "AwsSecretsManagerProvider"
    assert PostgresLeaseLockProvider.__name__ == "PostgresLeaseLockProvider"
    assert S3ArtifactStore.__name__ == "S3ArtifactStore"
