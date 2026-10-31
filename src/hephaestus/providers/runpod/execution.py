"""RunPod Pod lifecycle adapter using the public REST API."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from urllib.parse import urlencode, quote
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
    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


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
                f"RunPod API request failed with HTTP {exc.code}", status_code=exc.code
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
            "cpu": {"id": "cpu3c", "vcpuCount": 2},
            "cloud": "SECURE",
            "dataCenterIds": [self.config.datacenter_id],
            "image": image_name,
            "disk": 10,
            "mounts": {"network": [{"volumeId": self.config.network_volume_id, "path": "/workspace"}]},
            "entrypoint": [],
            "cmd": ["bash", "-lc", "sleep 300"],
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
        if interruptible:
            raise ValueError("REST v2 does not expose interruptible Pod creation; refusing a silent substitution")

        body: dict[str, object] = {
            "name": name,
            "gpu": {"id": normalized_gpu_ids[0], "count": 1},
            "cloud": "SECURE",
            "dataCenterIds": [self.config.datacenter_id],
            "image": image_name,
            "disk": container_disk_in_gb,
            "mounts": {"network": [{"volumeId": self.config.network_volume_id, "path": "/workspace"}]},
            "entrypoint": [],
            "cmd": list(docker_start_cmd),
        }
        if env:
            body["env"] = {str(key): str(value) for key, value in env.items()}
        return self._create_pod(body, operation_id=operation_id, gpu_type_ids=normalized_gpu_ids)

    def _create_pod(self, body: dict[str, object], *, operation_id: str | None = None,
                    gpu_type_ids: list[str] | None = None) -> dict[str, Any]:
        if not body.get("name") or self.maximum_reconciliation_attempts < 1:
            raise ValueError("a stable resource name and positive reconciliation limit are required")
        key = hashlib.sha256(canonical_bytes([self.config.api_base_url,
            self.config.network_volume_id, operation_id or body["name"]])).hexdigest()
        request_hash = hashlib.sha256(canonical_bytes({"body": body, "gpu_allowlist": gpu_type_ids})).hexdigest()
        resource_name = str(body["name"])[:140] + "-" + key[:32]
        intent = ExternalOperation("runpod-"+key, "runpod", request_hash, resource_name, "submitting")
        markers = {"HEPHAESTUS_OPERATION_ID":key, "HEPHAESTUS_REQUEST_SHA256":request_hash}
        environment = dict(body.get("env") or {})
        if set(environment).intersection(markers):
            raise ValueError("caller cannot override provider-operation identity markers")
        submitted = {**body, "name":resource_name, "env":{**environment, **markers}}
        with single_writer(self.state_root, "runpod"):
            store = ExternalOperationStore(SQLiteStateRepository(self.state_root / "runpod-operations.sqlite3"))
            legacy_key = hashlib.sha256(canonical_bytes(["https://rest.runpod.io/v1",
                self.config.network_volume_id, operation_id or body["name"]])).hexdigest()
            if store.get("runpod-" + legacy_key) is not None:
                raise AmbiguousRunPodOperation("legacy v1 intent exists; reconcile its resource before starting a v2 operation", resumable=False)
            existing = store.get(intent.operation_id)
            if existing is not None:
                if existing.request_sha256 != request_hash:
                    raise ValueError("RunPod operation retry changed its request")
                if existing.status == "blocked":
                    raise AmbiguousRunPodOperation("RunPod operation needs explicit reconciliation review", resumable=False)
                if existing.status in {"terminating", "terminated"}:
                    raise AmbiguousRunPodOperation("RunPod operation has entered termination; allocation cannot be repeated", resumable=False)
                if existing.external_id:
                    pod = self.get_pod(existing.external_id)
                    self._validate_reconciled_pod(pod, submitted, gpu_type_ids)
                    return pod
                if existing.reconciliation_attempts >= self.maximum_reconciliation_attempts:
                    store.record(replace(existing, status="blocked"))
                    raise AmbiguousRunPodOperation("RunPod operation exhausted its reconciliation budget", resumable=False)
                attempts = existing.reconciliation_attempts + 1
                pending = replace(existing, status="uncertain", reconciliation_attempts=attempts)
                store.record(pending)  # Bound reconciliation attempts, including interrupted GETs.
                # v2 has no name filter. Complete cursor discovery is necessary
                # even after a match, because another page may contain a duplicate.
                try:
                    matches = [pod for pod in self.list_pods() if pod.get("name") == resource_name]
                except RunPodApiError as exc:
                    raise AmbiguousRunPodOperation("provider discovery failed; allocation remains unresolved") from exc
                if len(matches) != 1:
                    store.record(replace(pending, status="blocked" if attempts >= self.maximum_reconciliation_attempts else "uncertain"))
                    raise AmbiguousRunPodOperation("provider acceptance is unresolved; duplicate allocation refused")
                pod = matches[0]
                self._validate_reconciled_pod(pod, submitted, gpu_type_ids)
                try:
                    store.record(replace(pending, status="accepted", external_id=str(pod["id"])))
                except Exception as exc:
                    raise AmbiguousRunPodOperation("provider resource was found but its receipt could not be persisted") from exc
                return pod
            if gpu_type_ids and len(gpu_type_ids) > 1:
                submitted["gpu"] = {"id": self.select_gpu(gpu_type_ids), "count": 1}
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

    def _validate_reconciled_pod(self, pod: dict, request: dict, gpu_type_ids: list[str] | None = None) -> None:
        env = pod.get("env") or {}
        gpu = pod.get("gpu") or {}
        if (not isinstance(gpu, dict) or not pod.get("id") or pod.get("name") != request["name"] or
                not isinstance(env, dict) or any(env.get(k) != v for k,v in request["env"].items()) or
                pod.get("mounts") != request["mounts"] or
                pod.get("dataCenterId") != self.config.datacenter_id or
                pod.get("cloud") != "SECURE" or pod.get("disk") != request["disk"] or
                pod.get("image") != request["image"] or pod.get("cmd") != request["cmd"] or
                pod.get("entrypoint") != request["entrypoint"] or
                (gpu_type_ids and (gpu.get("count") != 1 or gpu.get("id") not in gpu_type_ids)) or
                ("cpu" in request and (not isinstance(pod.get("cpu"), dict) or
                 pod["cpu"].get("id") != request["cpu"]["id"] or
                 pod["cpu"].get("vcpuCount") != request["cpu"]["vcpuCount"]))):
            raise AmbiguousRunPodOperation("discovered provider resource does not match durable operation intent")

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        status, payload = self._request("GET", f"pods/{quote(pod_id, safe='')}")
        if status != 200 or not isinstance(payload, dict):
            raise RunPodApiError(f"unexpected get-Pod response status: {status}", status_code=status)
        if payload.get("id") != pod_id:
            raise RunPodApiError("get-Pod response has a mismatched resource identity")
        return payload

    def delete_pod(self, pod_id: str) -> None:
        if not pod_id:
            raise ValueError("Pod identity is required")
        with single_writer(self.state_root, "runpod"):
            store = ExternalOperationStore(SQLiteStateRepository(self.state_root / "runpod-operations.sqlite3"))
            operation = store.for_resource(pod_id)
            if operation is None:
                digest = hashlib.sha256(canonical_bytes([self.config.api_base_url, pod_id])).hexdigest()
                operation = ExternalOperation("runpod-delete-"+digest, "runpod", digest, pod_id, "terminating", external_id=pod_id)
            if operation.status == "terminated":
                return
            store.record(replace(operation, status="terminating"))
            try:
                status, _ = self._request("DELETE", f"pods/{quote(pod_id, safe='')}")
            except RunPodApiError as exc:
                if exc.status_code != 404:
                    raise
                status = 404
            if status not in {204, 404}:
                raise RunPodApiError(f"unexpected delete-Pod response status: {status}", status_code=status)
            # A repeated DELETE after a lost response is safe. A 404 confirms
            # absence; network errors retain the terminating record for retry.
            store.record(replace(operation, status="terminated"))

    @staticmethod
    def is_ready(pod: dict[str, Any]) -> bool:
        return pod.get("status") == "RUNNING"

    def list_pods(self) -> list[dict[str, Any]]:
        pods: list[dict[str, Any]] = []
        cursor: str | None = None
        seen: set[str] = set()
        for _ in range(100):
            query = urlencode({"limit": 1000, **({"cursor": cursor} if cursor else {})})
            status, page = self._request("GET", "pods?" + query)
            if status != 200 or not isinstance(page, dict) or not isinstance(page.get("pods"), list):
                raise AmbiguousRunPodOperation("provider resource discovery is unavailable or ambiguous")
            if not all(isinstance(p, dict) and isinstance(p.get("id"), str) for p in page["pods"]):
                raise AmbiguousRunPodOperation("provider returned malformed Pod identities")
            pods.extend(page["pods"])
            pagination = page.get("pagination")
            if not isinstance(pagination, dict) or not isinstance(pagination.get("hasNextPage"), bool):
                raise AmbiguousRunPodOperation("provider pagination metadata is missing")
            if not pagination["hasNextPage"]:
                return pods
            cursor = pagination.get("nextCursor")
            if not isinstance(cursor, str) or not cursor or cursor in seen:
                raise AmbiguousRunPodOperation("provider returned an invalid pagination cursor")
            seen.add(cursor)
        raise AmbiguousRunPodOperation("provider discovery exceeded bounded page limit")

    def select_gpu(self, allowed: list[str]) -> str:
        status, catalog = self._request("GET", "catalog/datacenters/" + quote(self.config.datacenter_id, safe="") + "?include=GPU_AVAILABILITY")
        if status != 200 or not isinstance(catalog, dict) or catalog.get("id") != self.config.datacenter_id:
            raise RunPodApiError("GPU placement catalog unavailable; no allocation attempted")
        available = {item.get("id") for item in catalog.get("gpuAvailability", [])
                     if isinstance(item, dict) and item.get("availability") in {"LOW", "MEDIUM", "HIGH"}}
        for gpu_id in allowed:
            if gpu_id in available:
                return gpu_id
        raise RunPodApiError("No allowed GPU available in the pinned datacenter; no allocation attempted")
