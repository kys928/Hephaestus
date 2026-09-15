#!/usr/bin/env python3
"""Launch and independently verify the S3-reviewed targeted volume cleanup."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import launch_first_bounded_scientific_training as base
from runpod_capacity_selection import VERIFIED_GPU_IDS
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig, RunPodExecutionAdapter

VOLUME_ID = "cviwpryzao"
DATACENTER_ID = "EU-CZ-1"
# Cleanup uses no CUDA functionality, so do not exclude Blackwell or other GPUs
# merely because a science image is pinned to an older CUDA runtime.
IMAGE = "ubuntu:24.04"
MANIFEST_PATH = Path("configs/maintenance/network_volume_targeted_cleanup_v1.json")
MAX_SECONDS = 3600
POLL_SECONDS = 5

# The first entry is known-valid because the live elasticity run was allocated
# on this exact GPU type.  The remaining extra routes were already approved for
# Hephaestus' >=48GB runtime envelope; the schema-verified legacy pool follows.
MAINTENANCE_EXTRA_GPU_IDS = (
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA RTX A6000",
)


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prefix_state(client: Any, prefix: str) -> dict[str, int]:
    count = 0
    total = 0
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": VOLUME_ID, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        page = client.list_objects_v2(**kwargs)
        for row in page.get("Contents", []) or []:
            count += 1
            total += int(row.get("Size") or 0)
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
        if not token:
            raise RuntimeError(f"S3 prefix pagination truncated without token: {prefix}")
    return {"object_count": count, "bytes": total}


def _pod_shell() -> str:
    return r'''set -Eeuo pipefail
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates coreutils python3
rm -rf /opt/hephaestus-src
git clone --filter=blob:none https://github.com/kys928/Hephaestus.git /opt/hephaestus-src
cd /opt/hephaestus-src
git checkout "$HEPHAESTUS_REPO_SHA"
python3 -m py_compile scripts/run_network_volume_targeted_cleanup_v1.py
python3 scripts/run_network_volume_targeted_cleanup_v1.py
'''


def _create_with_maintenance_capacity(execution: RunPodExecutionAdapter, body_factory) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Try Blackwell/Ada/A6000 routes plus the schema-verified legacy pool."""
    routes: list[list[str]] = [[gpu] for gpu in MAINTENANCE_EXTRA_GPU_IDS]
    routes.append(list(VERIFIED_GPU_IDS))
    observations: list[dict[str, Any]] = []
    last_error: BaseException | None = None
    for round_index in range(1, 13):
        for gpu_ids in routes:
            observation: dict[str, Any] = {
                "round": round_index,
                "gpu_type_ids": gpu_ids,
                "queried_at": _now(),
            }
            observations.append(observation)
            try:
                pod = execution._create_pod(body_factory(gpu_ids))
                observation["status"] = "created"
                observation["pod_id"] = pod.get("id")
                observation["gpu"] = pod.get("gpu")
                return pod, observations
            except BaseException as exc:
                last_error = exc
                observation["status"] = "failed"
                observation["error"] = f"{type(exc).__name__}: {exc}"
                lowered = str(exc).lower()
                if "402" in lowered or "insufficient funds" in lowered or "insufficient balance" in lowered:
                    raise
        if round_index < 12:
            time.sleep(10)
    raise RuntimeError(f"RunPod maintenance Pod capacity exhausted across all routes: {last_error}") from last_error


def main() -> int:
    _required("RUNPOD_API_KEY")
    repo_sha = _required("GITHUB_SHA")
    workflow_run_id = _required("GITHUB_RUN_ID")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["volume_id"] != VOLUME_ID or manifest["datacenter_id"] != DATACENTER_ID:
        raise RuntimeError("targeted cleanup manifest volume/datacenter mismatch")
    targets = manifest["delete_exact_prefixes"]
    expected_prefixes = {str(row["s3_prefix"]) for row in targets}
    hard_expected = {
        "hephaestus/scientific/v1/model_cache/huggingface/",
        "hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34104954995/hf_cache/",
    }
    if expected_prefixes != hard_expected:
        raise RuntimeError("targeted cleanup S3 prefix set differs from hard launcher allowlist")

    cleanup_run_id = f"network-volume-targeted-cleanup-v1-{workflow_run_id}"
    execution = RunPodExecutionAdapter(RunPodConfig.from_env(), EnvironmentSecretsProvider())
    client = base.s3_client()
    client.head_bucket(Bucket=VOLUME_ID)
    pod_id: str | None = None
    launcher: dict[str, Any] = {
        "launcher_version": "network-volume-targeted-cleanup-launcher.v2",
        "started_at": _now(),
        "cleanup_run_id": cleanup_run_id,
        "repo_sha": repo_sha,
        "volume_id": VOLUME_ID,
        "datacenter_id": DATACENTER_ID,
        "protected_active_run_id": manifest["protected_active_run_id"],
        "inventory_basis": manifest["basis"],
        "target_prefixes": sorted(expected_prefixes),
        "maintenance_extra_gpu_type_ids": list(MAINTENANCE_EXTRA_GPU_IDS),
        "schema_verified_fallback_gpu_type_ids": list(VERIFIED_GPU_IDS),
        "container_image": IMAGE,
    }
    try:
        launcher["s3_before"] = {prefix: _prefix_state(client, prefix) for prefix in sorted(expected_prefixes)}
        for row in targets:
            observed = launcher["s3_before"][row["s3_prefix"]]["bytes"]
            expected = int(row["s3_observed_bytes"])
            if observed != expected:
                raise RuntimeError(
                    f"S3 target changed since reviewed inventory: {row['s3_prefix']} observed={observed} expected={expected}"
                )

        def body_factory(gpu_ids: list[str]) -> dict[str, Any]:
            return {
                "name": f"hephaestus-{cleanup_run_id}"[:180],
                "computeType": "GPU",
                "gpuCount": 1,
                "gpuTypeIds": gpu_ids,
                "gpuTypePriority": "availability",
                "cloudType": "SECURE",
                "dataCenterIds": [DATACENTER_ID],
                "dataCenterPriority": "custom",
                "imageName": IMAGE,
                "containerDiskInGb": 20,
                "networkVolumeId": VOLUME_ID,
                "volumeMountPath": "/workspace",
                "dockerStartCmd": ["bash", "-lc", _pod_shell()],
                "interruptible": False,
                "env": {
                    "HEPHAESTUS_TARGETED_CLEANUP_RUN_ID": cleanup_run_id,
                    "HEPHAESTUS_REPO_SHA": repo_sha,
                },
            }

        pod, capacity = _create_with_maintenance_capacity(execution, body_factory)
        launcher["capacity_selection"] = capacity
        pod_id = str(pod["id"])
        launcher["pod_id"] = pod_id
        launcher["gpu"] = pod.get("gpu")

        report_key = f"hephaestus/scientific/v1/maintenance/targeted_storage_cleanup/{cleanup_run_id}/report.json"
        deadline = time.monotonic() + MAX_SECONDS
        report_raw: bytes | None = None
        while time.monotonic() < deadline:
            report_raw = base.maybe_read_key(client, report_key)
            if report_raw is not None:
                break
            time.sleep(POLL_SECONDS)
        if report_raw is None:
            raise TimeoutError(f"targeted cleanup report did not appear within {MAX_SECONDS} seconds")
        report = json.loads(report_raw.decode("utf-8"))
        Path("targeted_cleanup_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if report.get("status") != "completed" or report.get("deleted_target_count") != 2:
            raise RuntimeError(f"targeted cleanup worker did not delete both reviewed targets: {report.get('status')}")

        s3_after: dict[str, dict[str, int]] = {}
        verify_deadline = time.monotonic() + 180
        while True:
            s3_after = {prefix: _prefix_state(client, prefix) for prefix in sorted(expected_prefixes)}
            if all(state["object_count"] == 0 and state["bytes"] == 0 for state in s3_after.values()):
                break
            if time.monotonic() >= verify_deadline:
                raise RuntimeError(f"deleted cache prefixes still visible through S3: {s3_after}")
            time.sleep(5)

        active_prefix = f"hephaestus/scientific/v1/adaptation_elasticity/{manifest['protected_active_run_id']}/"
        active_state = _prefix_state(client, active_prefix)
        if active_state["object_count"] <= 0:
            raise RuntimeError("protected elasticity evidence disappeared from S3")
        old_proof_prefix = "hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34104954995/"
        old_proof_state = _prefix_state(client, old_proof_prefix)
        if old_proof_state["object_count"] <= 0:
            raise RuntimeError("old proof evidence disappeared with its cache; expected surrounding evidence to remain")

        verification = {
            "verification_version": "network-volume-targeted-cleanup-verification.v1",
            "verified_at": _now(),
            "s3_before": launcher["s3_before"],
            "s3_after": s3_after,
            "active_protected_prefix": active_prefix,
            "active_protected_state": active_state,
            "preserved_old_proof_prefix": old_proof_prefix,
            "preserved_old_proof_state": old_proof_state,
            "report_key": report_key,
            "reported_free_space_gained_bytes": report.get("free_space_gained_bytes"),
            "reported_deleted_apparent_bytes": report.get("deleted_apparent_bytes"),
            "s3_bytes_removed": sum(state["bytes"] for state in launcher["s3_before"].values()),
        }
        Path("targeted_cleanup_verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        launcher["status"] = "verified"
        launcher["verification"] = verification
        print("NETWORK_VOLUME_TARGETED_CLEANUP_LAUNCH_JSON " + json.dumps({
            "status": "verified",
            "pod_id": pod_id,
            "s3_bytes_removed": verification["s3_bytes_removed"],
            "reported_free_space_gained_bytes": verification["reported_free_space_gained_bytes"],
            "active_protected_objects": active_state["object_count"],
            "preserved_old_proof_objects": old_proof_state["object_count"],
        }, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        launcher["status"] = "failed"
        launcher["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if pod_id:
            launcher["teardown"] = base.delete_pod(execution, pod_id)
        else:
            launcher["teardown"] = {"deleted": False, "not_created": True}
        launcher["completed_at"] = _now()
        Path("targeted_cleanup_launcher.json").write_text(json.dumps(launcher, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
