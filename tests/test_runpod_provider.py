from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hephaestus.infrastructure.secrets import InjectedSecretsProvider
from hephaestus.providers.runpod import (
    RunPodConfig,
    RunPodExecutionAdapter,
    RunPodNetworkVolumeStorage,
)


def _config() -> RunPodConfig:
    return RunPodConfig.from_env(
        {
            "RUNPOD_DATACENTER_ID": "EU-CZ-1",
            "RUNPOD_NETWORK_VOLUME_ID": "cviwpryzao",
            "RUNPOD_S3_ENDPOINT_URL": "https://s3api-eu-cz-1.runpod.io/",
            "RUNPOD_API_BASE_URL": "https://rest.runpod.io/v1",
        }
    )


def _secrets() -> InjectedSecretsProvider:
    values = {
        "RUNPOD_API_KEY": "api-test",
        "RUNPOD_S3_ACCESS_KEY_ID": "access-test",
        "RUNPOD_S3_SECRET_ACCESS_KEY": "secret-test",
    }
    return InjectedSecretsProvider("environment", values.__getitem__)


def test_runpod_config_preserves_exact_repository_contract() -> None:
    config = _config()
    assert config.datacenter_id == "EU-CZ-1"
    assert config.network_volume_id == "cviwpryzao"
    assert config.s3_endpoint_url == "https://s3api-eu-cz-1.runpod.io"
    assert config.api_base_url == "https://rest.runpod.io/v1"
    assert config.api_key_ref.key == "RUNPOD_API_KEY"
    assert config.s3_access_key_id_ref.key == "RUNPOD_S3_ACCESS_KEY_ID"
    assert config.s3_secret_access_key_ref.key == "RUNPOD_S3_SECRET_ACCESS_KEY"


@dataclass
class FakeS3Client:
    calls: list[tuple[str, dict[str, object]]] = field(default_factory=list)

    def head_bucket(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("head_bucket", kwargs))
        return {}

    def list_objects_v2(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("list_objects_v2", kwargs))
        return {
            "Contents": [
                {"Key": "datasets/train.jsonl", "Size": 123},
                {"Key": "checkpoints/model.pt", "Size": 456},
            ],
            "IsTruncated": False,
        }


def test_network_volume_adapter_targets_configured_volume() -> None:
    client = FakeS3Client()
    storage = RunPodNetworkVolumeStorage(client=client, config=_config())
    storage.head_volume()
    listing = storage.list_objects(max_items=50)
    assert listing.volume_id == "cviwpryzao"
    assert [item.key for item in listing.objects] == [
        "datasets/train.jsonl",
        "checkpoints/model.pt",
    ]
    assert client.calls[0] == ("head_bucket", {"Bucket": "cviwpryzao"})
    assert client.calls[1][1] == {
        "Bucket": "cviwpryzao",
        "Prefix": "",
        "MaxKeys": 50,
    }


@dataclass
class FakeTransport:
    calls: list[dict[str, Any]] = field(default_factory=list)

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_body: dict[str, object] | None = None,
    ) -> tuple[int, dict[str, Any] | None]:
        self.calls.append(
            {"method": method, "url": url, "headers": headers, "json_body": json_body}
        )
        if method == "POST":
            return 201, {
                "id": "pod-test",
                "desiredStatus": "RUNNING",
                "networkVolume": {"id": "cviwpryzao"},
            }
        if method == "DELETE":
            return 204, None
        raise AssertionError(f"unexpected method: {method}")


def test_execution_adapter_creates_pinned_cpu_pod_and_deletes_it() -> None:
    transport = FakeTransport()
    execution = RunPodExecutionAdapter(
        config=_config(), secrets=_secrets(), transport=transport
    )
    pod = execution.create_disposable_cpu_pod(name="smoke")
    assert pod["id"] == "pod-test"
    create = transport.calls[0]
    assert create["url"] == "https://rest.runpod.io/v1/pods"
    assert create["headers"]["Authorization"] == "Bearer api-test"
    body = create["json_body"]
    assert body["computeType"] == "CPU"
    assert body["cpuFlavorIds"] == ["cpu3c"]
    assert body["dataCenterIds"] == ["EU-CZ-1"]
    assert body["networkVolumeId"] == "cviwpryzao"
    assert body["volumeMountPath"] == "/workspace"

    execution.delete_pod("pod-test")
    assert transport.calls[1]["method"] == "DELETE"
    assert transport.calls[1]["url"] == "https://rest.runpod.io/v1/pods/pod-test"
