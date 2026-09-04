#!/usr/bin/env python3
"""Provide a bounded current RunPod GPU allowlist for availability-priority Pod scheduling.

The GraphQL catalog endpoint is blocked from GitHub-hosted runners by RunPod's
edge policy (HTTP 403 / error 1010). The REST v1 Pod schema itself publishes the
accepted GPU enum and supports ``gpuTypePriority=availability``. We therefore
let the Pod scheduler determine live capacity inside the fixed EU-CZ-1 Secure
Cloud datacenter. This changes execution routing only, never scientific inputs.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

DATACENTER = "EU-CZ-1"

# Current RunPod REST Pod GPU enum, restricted to NVIDIA GPUs with >=16 GB and
# CUDA-generation compatibility appropriate for the pinned CUDA 12.6 image.
# Prefer the RTX 4090 used by the first run; availability priority lets RunPod
# choose another listed type only when necessary.
OFFICIAL_GPU_IDS = (
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA L4",
    "NVIDIA RTX A5000",
    "NVIDIA RTX A6000",
    "NVIDIA A40",
    "NVIDIA L40",
    "NVIDIA L40S",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA RTX 5000 Ada Generation",
    "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA A30",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA H100 PCIe",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 NVL",
    "NVIDIA H200",
    "NVIDIA GeForce RTX 4080 SUPER",
    "NVIDIA GeForce RTX 4080",
)


def scheduler_evidence(attempt: int) -> dict[str, object]:
    return {
        "attempt": attempt,
        "queried_at": datetime.now(timezone.utc).isoformat(),
        "selection_source": "runpod_rest_v1_official_gpu_enum_with_scheduler_availability",
        "catalog_graphql_from_github_runner": "blocked_http_403_error_1010",
        "datacenter_id": DATACENTER,
        "cloud_type": "SECURE",
        "gpu_count": 1,
        "gpu_type_priority": "availability",
        "ordered_gpu_type_ids": list(OFFICIAL_GPU_IDS),
    }


def create_with_capacity_retries(
    create_once,
    *,
    attempts: int = 6,
    delay_seconds: float = 10.0,
):
    observations: list[dict[str, object]] = []
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        observation = scheduler_evidence(attempt)
        observations.append(observation)
        try:
            pod = create_once(list(OFFICIAL_GPU_IDS))
            observation["create_status"] = "created"
            observation["created_pod_id"] = pod.get("id") if isinstance(pod, dict) else None
            if isinstance(pod, dict):
                gpu = pod.get("gpu")
                if isinstance(gpu, dict):
                    observation["allocated_gpu"] = {
                        "id": gpu.get("id"),
                        "displayName": gpu.get("displayName"),
                        "count": gpu.get("count"),
                    }
            return pod, observations
        except BaseException as exc:
            last_error = exc
            observation["create_status"] = "failed"
            observation["create_error"] = f"{type(exc).__name__}: {exc}"
            lowered = str(exc).lower()
            if "402" in lowered or "insufficient" in lowered:
                raise
        if attempt < attempts:
            time.sleep(delay_seconds)
    raise RuntimeError(f"RunPod availability-priority Pod creation exhausted retries: {last_error}") from last_error
