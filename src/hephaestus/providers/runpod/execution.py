"""RunPod Pod lifecycle adapter using the public REST API."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from urllib.parse import urlencode
from dataclasses import replace
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from hephaestus.infrastructure.secrets import SecretsProvider

from .config import RunPodConfig
from hephaestus.schemas.external_operation import ExternalOperation, ExternalOperationPending
from hephaestus.state.external_operation_store import ExternalOperationStore
from hephaestus.state.single_writer import single_writer
from hephaestus.storage.sqlite import SQLiteStateRepository
from hephaestus.storage.verified import canonical_bytes


class RunPodApiError(RuntimeError):
    pass


class AmbiguousRunPodOperation(RunPodApiError, ExternalOperationPending):
    """Allocation outcome is unknown. A retry must reconcile, never POST again."""

    def __init__(self, message: str, *, resumable: bool = True):
        ExternalOperationPending.__init__(self, message, resumable=resumable)


class RunPodHttpTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_body: dict[str, object] | None = None,
    ) -> tuple[int, dict[str, Any] | list[dict[str, Any]] | None]: ...


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
    ) -> tuple[int, dict[str, Any] | list[dict[str, Any]] | None]:
        data = None if json_body is None else json.dumps(json_body).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
                payload = json.loads(raw) if raw else None
                return int(response.status), payload
        except urllib.error.HTTPError as exc:
            raise RunPodApiError(
                f"RunPod API request failed with HTTP {exc.code}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RunPodApiError("RunPod API request failed at the transport layer") from exc


@dataclass(slots=True)
class RunPodExecutionAdapter:
    config: RunPodConfig
    secrets: SecretsProvider = field(repr=False)
    transport: RunPodHttpTransport = field(default_factory=UrllibRunPodTransport, repr=False)
    # This must be mounted durable storage for controller restart recovery.
    state_root: Path = Path("state")
    maximum_reconciliation_attempts: int = 3

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, object] | None = None,
    ) -> tuple[int, dict[str, Any] | list[dict[str, Any]] | None]:
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
        operation_id: str | None = None,
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
        return self._create_pod(body, operation_id=operation_id)

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
        operation_id: str | None = None,
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
        return self._create_pod(body, operation_id=operation_id)

    def _create_pod(self, body: dict[str, object], *, operation_id: str | None = None) -> dict[str, Any]:
        if not body.get("name") or self.maximum_reconciliation_attempts < 1:
            raise ValueError("a stable resource name and positive reconciliation limit are required")
        key = hashlib.sha256(canonical_bytes([self.config.api_base_url,
            self.config.network_volume_id, operation_id or body["name"]])).hexdigest()
        request_hash = hashlib.sha256(canonical_bytes(body)).hexdigest()
        resource_name = str(body["name"])[:140] + "-" + key[:32]
        intent = ExternalOperation("runpod-"+key, "runpod", request_hash, resource_name, "submitting")
        markers = {"HEPHAESTUS_OPERATION_ID":key, "HEPHAESTUS_REQUEST_SHA256":request_hash}
        environment = dict(body.get("env") or {})
        if set(environment).intersection(markers):
            raise ValueError("caller cannot override provider-operation identity markers")
        submitted = {**body, "name":resource_name, "env":{**environment, **markers}}
        with single_writer(self.state_root, "runpod"):
            store = ExternalOperationStore(SQLiteStateRepository(self.state_root / "runpod-operations.sqlite3"))
            existing = store.get(intent.operation_id)
            if existing is not None:
                if existing.request_sha256 != request_hash:
                    raise ValueError("RunPod operation retry changed its request")
                if existing.status == "blocked":
                    raise AmbiguousRunPodOperation("RunPod operation needs explicit reconciliation review", resumable=False)
                if existing.external_id:
                    pod = self.get_pod(existing.external_id)
                    self._validate_reconciled_pod(pod, submitted)
                    return pod
                if existing.reconciliation_attempts >= self.maximum_reconciliation_attempts:
                    store.record(replace(existing, status="blocked"))
                    raise AmbiguousRunPodOperation("RunPod operation exhausted its reconciliation budget", resumable=False)
                attempts = existing.reconciliation_attempts + 1
                pending = replace(existing, status="uncertain", reconciliation_attempts=attempts)
                store.record(pending)  # Bound reconciliation attempts, including interrupted GETs.
                query = urlencode({"name":resource_name, "includeNetworkVolume":"true", "includeMachine":"true"})
                status, response = self._request("GET", "pods?"+query)
                if status != 200 or not isinstance(response, list) or not all(isinstance(x, dict) for x in response):
                    raise AmbiguousRunPodOperation("provider resource discovery is unavailable or ambiguous")
                matches = [pod for pod in response if pod.get("name") == resource_name]
                if len(matches) != 1:
                    store.record(replace(pending, status="blocked" if attempts >= self.maximum_reconciliation_attempts else "uncertain"))
                    raise AmbiguousRunPodOperation("provider acceptance is unresolved; duplicate allocation refused")
                pod = matches[0]
                self._validate_reconciled_pod(pod, submitted)
                try:
                    store.record(replace(pending, status="accepted", external_id=str(pod["id"])))
                except Exception as exc:
                    raise AmbiguousRunPodOperation("provider resource was found but its receipt could not be persisted") from exc
                return pod
            # A crash after this durable intent can only enter reconciliation.
            store.record(intent)
            try:
                status, payload = self._request("POST", "pods", submitted)
                if status != 201 or not isinstance(payload, dict) or not isinstance(payload.get("id"), str) or not payload["id"]:
                    raise RunPodApiError("create-Pod response did not confirm an allocation identity")
            except Exception as exc:
                store.record(replace(intent, status="uncertain"))
                raise AmbiguousRunPodOperation("provider acceptance is unknown; reconcile this operation before retrying") from exc
            try:
                store.record(replace(intent, status="accepted", external_id=payload["id"]))
            except Exception as exc:
                raise AmbiguousRunPodOperation("provider resource was accepted but its receipt could not be persisted") from exc
            return payload

    def _validate_reconciled_pod(self, pod: dict, request: dict) -> None:
        volume = pod.get("networkVolume") or {}
        env = pod.get("env") or {}
        gpu = pod.get("gpu") or {}
        image = pod.get("imageName") or pod.get("image")
        if (not isinstance(volume, dict) or not isinstance(gpu, dict) or not pod.get("id") or pod.get("name") != request["name"] or
                not isinstance(env, dict) or any(env.get(k) != v for k,v in request["env"].items()) or
                volume.get("id", pod.get("networkVolumeId")) != self.config.network_volume_id or
                volume.get("dataCenterId", self.config.datacenter_id) != self.config.datacenter_id or
                image != request["imageName"] or pod.get("dockerStartCmd") != request["dockerStartCmd"] or
                pod.get("volumeMountPath") != request["volumeMountPath"] or
                (request["computeType"] == "GPU" and gpu.get("count", pod.get("gpuCount")) != 1)):
            raise AmbiguousRunPodOperation("discovered provider resource does not match durable operation intent")

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        status, payload = self._request("GET", f"pods/{pod_id}")
        if status != 200 or not isinstance(payload, dict):
            raise RunPodApiError(f"unexpected get-Pod response status: {status}")
        return payload

    def delete_pod(self, pod_id: str) -> None:
        status, _ = self._request("DELETE", f"pods/{pod_id}")
        if status != 204:
            raise RunPodApiError(f"unexpected delete-Pod response status: {status}")
