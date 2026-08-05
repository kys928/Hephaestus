from __future__ import annotations

import json

import pytest

from hephaestus.infrastructure.cli import main
from hephaestus.infrastructure.config import InfrastructureConfig, InfrastructureConfigError
from hephaestus.infrastructure.observability import JsonLineEventSink, StructuredEvent
from hephaestus.infrastructure.secrets import (
    EnvironmentSecretsProvider,
    SecretReference,
    SecretResolutionError,
)


def test_only_secret_reference_is_serializable_and_value_is_resolved_at_runtime() -> None:
    reference = SecretReference("environment", "MODEL_PROVIDER_TOKEN")
    provider = EnvironmentSecretsProvider(environ={"MODEL_PROVIDER_TOKEN": "do-not-persist"})

    assert reference.to_dict() == {
        "provider": "environment",
        "key": "MODEL_PROVIDER_TOKEN",
    }
    assert "do-not-persist" not in json.dumps(reference.to_dict())
    assert provider.resolve(reference) == "do-not-persist"
    with pytest.raises(SecretResolutionError):
        provider.resolve(SecretReference("vault", "MODEL_PROVIDER_TOKEN"))


def test_configuration_loads_explicit_roots_and_validates_lease() -> None:
    config = InfrastructureConfig.from_env(
        {
            "HEPHAESTUS_STATE_ROOT": "/state",
            "HEPHAESTUS_ARTIFACT_ROOT": "/artifacts",
            "HEPHAESTUS_WORKER_ID": "worker-1",
            "HEPHAESTUS_JOB_LEASE_SECONDS": "30",
        }
    )
    assert str(config.state_root) == "/state"
    assert str(config.artifact_root) == "/artifacts"
    assert config.lease_seconds == 30
    with pytest.raises(InfrastructureConfigError):
        InfrastructureConfig.from_env({"HEPHAESTUS_JOB_LEASE_SECONDS": "0"})


def test_structured_json_event_sink_emits_jsonl(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    sink = JsonLineEventSink(path)
    sink.emit(
        StructuredEvent.create(
            "health.checked", "test", entity_id="health", attributes={"ready": True}
        )
    )

    payload = json.loads(path.read_text())
    assert payload["event_type"] == "health.checked"
    assert payload["attributes"] == {"ready": True}


def test_health_entry_point_reports_ready_without_network(monkeypatch, tmp_path, capsys) -> None:
    state_root = tmp_path / "state"
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setenv("HEPHAESTUS_STATE_ROOT", str(state_root))
    monkeypatch.setenv("HEPHAESTUS_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("HEPHAESTUS_EVENT_LOG", str(state_root / "events.jsonl"))

    assert main(["health", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["live"] is True
    assert payload["ready"] is True
    assert state_root.joinpath("events.jsonl").exists()
