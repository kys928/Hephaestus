"""RunPod Pod lifecycle adapter using the public REST API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from hephaestus.infrastructure.secrets import SecretsProvider

from .config import RunPodConfig


class RunPodApiError(RuntimeError):
    pass


class RunPodHttpTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_body: dict[str, object] | None = None,
    ) -> tuple[int, dict[str, Any] | None]: ...


@dataclass(slots=True)
class UrllibRunPodTransport:
    timeout_seconds: float = 30.0

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_body: dict[str, object] | None = None,
    ) -> tuple[int, dict[str, Any] | None]:
        data = None if json_body is None else json.dumps(json_body).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
                payload = json.loads(raw) if raw else None
                return int(response.status), payload
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RunPodApiError(
                f"RunPod API request failed with HTTP {exc.code}: {raw}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RunPodApiError("RunPod API request failed at the transport layer") from exc


@dataclass(slots=True)
class RunPodExecutionAdapter:
    config: RunPodConfig
    secrets: SecretsProvider = field(repr=False)
    transport: RunPodHttpTransport = field(default_factory=UrllibRunPodTransport, repr=False)

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, object] | None = None,
    ) -> tuple[int, dict[str, Any] | None]:
        api_key = self.secrets.resolve(self.config.api_key_ref)
        headers = {"Authorization": f"Bearer {api_key}"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        return self.transport.request(
            method,
            f"{self.config.api_base_url}/{path.lstrip('/')}",
            headers=headers,
            json_body=body,
        )

    def create_disposable_cpu_pod(
        self,
        *,
        name: str = "hephaestus-runpod-smoke",
        image_name: str = "ubuntu:22.04",
    ) -> dict[str, Any]:
        body: dict[str, object] = {
            "name": name,
            "computeType": "CPU",
            "cpuFlavorIds": ["cpu3c"],
            "cpuFlavorPriority": "custom",
            "vcpuCount": 2,
            "cloudType": "SECURE",
            "dataCenterIds": [self.config.datacenter_id],
            "dataCenterPriority": "custom",
            "imageName": image_name,
            "containerDiskInGb": 10,
            "networkVolumeId": self.config.network_volume_id,
            "volumeMountPath": "/workspace",
            "dockerStartCmd": ["bash", "-lc", "sleep 300"],
            "interruptible": False,
        }
        return self._create_pod(body)

    def create_bounded_gpu_pod(
        self,
        *,
        name: str,
        image_name: str,
        gpu_type_ids: list[str],
        docker_start_cmd: list[str],
        env: dict[str, str] | None = None,
        container_disk_in_gb: int = 20,
        interruptible: bool = False,
    ) -> dict[str, Any]:
        """Create one GPU Pod pinned to this config's datacenter and Network Volume.

        The caller supplies only the bounded runtime command/environment and an
        ordered GPU allowlist. Volume identity, mount point, Secure Cloud, and
        single-GPU shape remain fixed by this adapter.
        """

        normalized_gpu_ids = [str(item).strip() for item in gpu_type_ids if str(item).strip()]
        if not normalized_gpu_ids:
            raise ValueError("at least one GPU type must be provided")
        if not docker_start_cmd or not all(str(item).strip() for item in docker_start_cmd):
            raise ValueError("docker_start_cmd must contain non-empty arguments")
        if container_disk_in_gb < 10:
            raise ValueError("container_disk_in_gb must be at least 10")

        body: dict[str, object] = {
            "name": name,
            "computeType": "GPU",
            "gpuCount": 1,
            "gpuTypeIds": normalized_gpu_ids,
            "gpuTypePriority": "custom",
            "cloudType": "SECURE",
            "dataCenterIds": [self.config.datacenter_id],
            "dataCenterPriority": "custom",
            "imageName": image_name,
            "containerDiskInGb": container_disk_in_gb,
            "networkVolumeId": self.config.network_volume_id,
            "volumeMountPath": "/workspace",
            "dockerStartCmd": list(docker_start_cmd),
            "interruptible": bool(interruptible),
        }
        if env:
            body["env"] = {str(key): str(value) for key, value in env.items()}
        return self._create_pod(body)

    def _create_pod(self, body: dict[str, object]) -> dict[str, Any]:
        status, payload = self._request("POST", "pods", body)
        if status != 201 or not isinstance(payload, dict):
            raise RunPodApiError(f"unexpected create-Pod response status: {status}")
        pod_id = payload.get("id")
        if not isinstance(pod_id, str) or not pod_id:
            raise RunPodApiError("create-Pod response did not contain a Pod id")
        return payload

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        status, payload = self._request("GET", f"pods/{pod_id}")
        if status != 200 or not isinstance(payload, dict):
            raise RunPodApiError(f"unexpected get-Pod response status: {status}")
        return payload

    def delete_pod(self, pod_id: str) -> None:
        status, _ = self._request("DELETE", f"pods/{pod_id}")
        if status != 204:
            raise RunPodApiError(f"unexpected delete-Pod response status: {status}")
