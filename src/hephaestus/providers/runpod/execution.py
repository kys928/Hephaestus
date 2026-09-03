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
