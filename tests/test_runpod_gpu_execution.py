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
        return 201, {"id": "pod-test", "desiredStatus": "RUNNING"}


def test_bounded_gpu_pod_is_pinned_to_configured_volume_and_datacenter() -> None:
    config = RunPodConfig.from_env({
        "RUNPOD_DATACENTER_ID": "EU-CZ-1",
        "RUNPOD_NETWORK_VOLUME_ID": "cviwpryzao",
        "RUNPOD_S3_ENDPOINT_URL": "https://s3api-eu-cz-1.runpod.io/",
        "RUNPOD_API_BASE_URL": "https://rest.runpod.io/v1",
    })
    transport = _Transport()
    execution = RunPodExecutionAdapter(config=config, secrets=_Secrets(), transport=transport)
    pod = execution.create_bounded_gpu_pod(
        name="scientific-run",
        image_name="pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime",
        gpu_type_ids=["Tesla T4", "NVIDIA RTX A4000"],
        docker_start_cmd=["bash", "-lc", "python run.py"],
        env={"HEPHAESTUS_RUN_ID": "run-1"},
    )
    assert pod["id"] == "pod-test"
    body = transport.calls[0]["json_body"]
    assert body["computeType"] == "GPU"
    assert body["gpuCount"] == 1
    assert body["gpuTypeIds"] == ["Tesla T4", "NVIDIA RTX A4000"]
    assert body["gpuTypePriority"] == "custom"
    assert body["cloudType"] == "SECURE"
    assert body["dataCenterIds"] == ["EU-CZ-1"]
    assert body["networkVolumeId"] == "cviwpryzao"
    assert body["volumeMountPath"] == "/workspace"
    assert body["dockerStartCmd"] == ["bash", "-lc", "python run.py"]
    assert body["env"] == {"HEPHAESTUS_RUN_ID": "run-1"}
    assert body["interruptible"] is False
