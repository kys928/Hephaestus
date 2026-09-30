#!/usr/bin/env python3
"""Launch a conservative RunPod maintenance pod on the Hephaestus Network Volume."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import launch_first_bounded_scientific_training as base
from runpod_capacity_selection import VERIFIED_GPU_IDS, create_with_capacity_retries
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig, RunPodExecutionAdapter

VOLUME_ID = "cviwpryzao"
DATACENTER_ID = "EU-CZ-1"
IMAGE = "pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime"
MAX_SECONDS = 3600
POLL_SECONDS = 5
PROTECTED_RUN_ID = "adaptation-elasticity-v1-34961824753"


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def pod_shell() -> str:
    return r'''set -Eeuo pipefail
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates coreutils
rm -rf /opt/hephaestus-src
git clone --filter=blob:none https://github.com/kys928/Hephaestus.git /opt/hephaestus-src
cd /opt/hephaestus-src
git checkout "$HEPHAESTUS_REPO_SHA"
python -m py_compile scripts/run_network_volume_cleanup_v1.py
python scripts/run_network_volume_cleanup_v1.py
'''


def main() -> int:
    required("RUNPOD_API_KEY")
    repo_sha = required("GITHUB_SHA")
    workflow_run_id = required("GITHUB_RUN_ID")
    cleanup_run_id = f"network-volume-cleanup-v1-{workflow_run_id}"
    execution = RunPodExecutionAdapter(RunPodConfig.from_env(), EnvironmentSecretsProvider())
    client = base.s3_client()
    client.head_bucket(Bucket=VOLUME_ID)
    pod_id: str | None = None
    launcher: dict[str, Any] = {
        "launcher_version": "network-volume-cleanup-launcher.v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "cleanup_run_id": cleanup_run_id,
        "repo_sha": repo_sha,
        "protected_run_id": PROTECTED_RUN_ID,
        "volume_id": VOLUME_ID,
        "datacenter_id": DATACENTER_ID,
        "gpu_type_ids": list(VERIFIED_GPU_IDS),
    }
    try:
        def create_once(gpu_ids: list[str]) -> dict[str, Any]:
            return execution._create_pod({
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
                "dockerStartCmd": ["bash", "-lc", pod_shell()],
                "interruptible": False,
                "env": {
                    "HEPHAESTUS_CLEANUP_RUN_ID": cleanup_run_id,
                    "HEPHAESTUS_PROTECTED_RUN_ID": PROTECTED_RUN_ID,
                    "HEPHAESTUS_REPO_SHA": repo_sha,
                },
            })

        pod, capacity = create_with_capacity_retries(create_once, attempts=12, delay_seconds=10.0)
        launcher["capacity_selection"] = capacity
        pod_id = str(pod["id"])
        launcher["pod_id"] = pod_id
        launcher["gpu"] = pod.get("gpu")
        key = f"hephaestus/scientific/v1/maintenance/storage_cleanup/{cleanup_run_id}/report.json"
        deadline = time.monotonic() + MAX_SECONDS
        report_raw: bytes | None = None
        while time.monotonic() < deadline:
            report_raw = base.maybe_read_key(client, key)
            if report_raw is not None:
                break
            time.sleep(POLL_SECONDS)
        if report_raw is None:
            raise TimeoutError(f"cleanup report did not appear within {MAX_SECONDS} seconds")
        report = json.loads(report_raw.decode("utf-8"))
        Path("storage_cleanup_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        launcher["status"] = "completed"
        launcher["report_key"] = key
        launcher["report"] = {
            "status": report.get("status"),
            "free_space_gained_bytes": report.get("free_space_gained_bytes"),
            "deleted_entry_count": report.get("deleted_entry_count"),
            "failure_count": len(report.get("failures", [])),
        }
        print("NETWORK_VOLUME_CLEANUP_LAUNCH_JSON " + json.dumps(launcher["report"], sort_keys=True), flush=True)
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
        launcher["completed_at"] = datetime.now(timezone.utc).isoformat()
        Path("storage_cleanup_launcher.json").write_text(json.dumps(launcher, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
