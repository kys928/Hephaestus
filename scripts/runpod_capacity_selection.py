#!/usr/bin/env python3
"""Compatibility launcher for ordered GPU preferences.

REST v2 takes one GPU ID. RunPodExecutionAdapter resolves these preferences
through the pinned datacenter catalog before recording an allocation intent.
An unresolved allocation always escapes this helper without another attempt.
"""
from __future__ import annotations

import time
from hephaestus.providers.runpod.execution import RunPodApiError, AmbiguousRunPodOperation
from datetime import datetime, timezone

DATACENTER = "EU-CZ-1"

# Historical preferences only; every request is checked against the v2 catalog. Blackwell consumer GPUs are deliberately omitted because
# the experiment image is pinned to CUDA 12.6. RTX 4090 remains first because it
# is the hardware used by the first controlled training run.
VERIFIED_GPU_IDS = (
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA L4",
    "NVIDIA A40",
    "NVIDIA L40",
    "NVIDIA L40S",
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
        "selection_source": "runpod_rest_v2_datacenter_catalog_selection",
        "datacenter_id": DATACENTER,
        "cloud_type": "SECURE",
        "gpu_count": 1,
        "gpu_type_priority": "ordered_catalog_availability",
        "ordered_gpu_type_ids": list(VERIFIED_GPU_IDS),
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
            pod = create_once(list(VERIFIED_GPU_IDS))
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
        except AmbiguousRunPodOperation:
            raise
        except RunPodApiError as exc:
            last_error = exc
            observation["create_status"] = "failed"
            observation["create_error"] = f"{type(exc).__name__}: {exc}"
            lowered = str(exc).lower()
            if "402" in lowered or "insufficient" in lowered:
                raise
        if attempt < attempts:
            time.sleep(delay_seconds)
    raise RuntimeError(f"RunPod availability-priority Pod creation exhausted retries: {last_error}") from last_error
