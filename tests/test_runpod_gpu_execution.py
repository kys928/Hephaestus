from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hephaestus.providers.runpod import RunPodConfig, RunPodExecutionAdapter


@dataclass
class _Secrets:
    provider_id: str = "environment"

    def resolve(self, reference):
        assert reference.provider == self.provider_id
        return "test-value"


@dataclass
class _Transport:
    calls: list[dict[str, Any]] = field(default_factory=list)

    def request(self, method, url, *, headers, json_body=None):
        self.calls.append({"method": method, "url": url, "headers": headers, "json_body": json_body})
        if method == "GET":
            return 200, {"id": "EU-CZ-1", "gpuAvailability": [{"id": "Tesla T4", "availability": "HIGH"}]}
        return 201, {"id": "pod-test", "status": "PROVISIONING"}


def test_bounded_gpu_pod_is_pinned_to_configured_volume_and_datacenter(tmp_path) -> None:
    config = RunPodConfig.from_env({
        "RUNPOD_DATACENTER_ID": "EU-CZ-1",
        "RUNPOD_NETWORK_VOLUME_ID": "cviwpryzao",
        "RUNPOD_S3_ENDPOINT_URL": "https://s3api-eu-cz-1.runpod.io/",
        "RUNPOD_API_BASE_URL": "https://api.runpod.io/v2",
    })
    transport = _Transport()
    execution = RunPodExecutionAdapter(config=config, secrets=_Secrets(), transport=transport, state_root=tmp_path)
    pod = execution.create_bounded_gpu_pod(
        name="scientific-run",
        image_name="pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime",
        gpu_type_ids=["Tesla T4", "NVIDIA RTX A4000"],
        docker_start_cmd=["bash", "-lc", "python run.py"],
        env={"HEPHAESTUS_RUN_ID": "run-1"},
    )
    assert pod["id"] == "pod-test"
    body = transport.calls[-1]["json_body"]
    assert body["gpu"] == {"id": "Tesla T4", "count": 1}
    assert body["cloud"] == "SECURE"
    assert body["dataCenterIds"] == ["EU-CZ-1"]
    assert body["mounts"] == {"network": [{"volumeId": "cviwpryzao", "path": "/workspace"}]}
    assert body["cmd"] == ["bash", "-lc", "python run.py"]
    assert body["env"]["HEPHAESTUS_RUN_ID"] == "run-1"
    assert len(body["env"]["HEPHAESTUS_OPERATION_ID"]) == 64
    assert len(body["env"]["HEPHAESTUS_REQUEST_SHA256"]) == 64
    assert "interruptible" not in body
    assert body["entrypoint"] == []
