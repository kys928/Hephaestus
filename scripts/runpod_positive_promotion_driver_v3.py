#!/usr/bin/env python3
"""Generic RunPod production-loop driver wrapper for promotion wave V3."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import launch_positive_promotion_proof_v3 as launcher_v3  # noqa: E402
import runpod_positive_promotion_driver as base  # noqa: E402

# V3 keeps the exact FP16 scientific candidates and frozen evaluation, but
# changes execution topology to two 24GB RTX 3090s. Transformers/Accelerate
# shards each 14B candidate across both devices; no quantization or CPU/disk
# offload is permitted. The container disk stores immutable model/cache files
# and runtime dependencies only; increasing it does not alter model execution.
V3_GPU_IDS = ("NVIDIA GeForce RTX 3090",)
V3_GPU_COUNT = 2
V3_PER_GPU_MEMORY_GB = 24
V3_AGGREGATE_GPU_MEMORY_GB = V3_GPU_COUNT * V3_PER_GPU_MEMORY_GB
V3_MAX_MEMORY_GIB_PER_GPU = 22
V3_MODEL_PARALLELISM = "transformers_device_map_balanced_fp16"
V3_CONTAINER_DISK_GB = 400


def _v3_create_with_capacity_retries(create_once, *, attempts: int = 60, delay_seconds: float = 10.0):
    """Persistently acquire the exact two-3090 topology without changing science."""
    observations: list[dict[str, object]] = []
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        observation: dict[str, object] = {
            "attempt": attempt,
            "selection_source": "v3_fp16_dual_3090_sharded",
            "per_gpu_memory_class_gb": V3_PER_GPU_MEMORY_GB,
            "aggregate_gpu_memory_gb": V3_AGGREGATE_GPU_MEMORY_GB,
            "max_memory_gib_per_gpu": V3_MAX_MEMORY_GIB_PER_GPU,
            "model_parallelism": V3_MODEL_PARALLELISM,
            "datacenter_id": base.launcher.DATACENTER_ID,
            "cloud_type": "SECURE",
            "gpu_count": V3_GPU_COUNT,
            "gpu_type_priority": "availability",
            "ordered_gpu_type_ids": list(V3_GPU_IDS),
        }
        observations.append(observation)
        try:
            pod = create_once(list(V3_GPU_IDS))
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
    raise RuntimeError(f"RunPod V3 dual-3090 Pod creation exhausted retries: {last_error}") from last_error


def _v3_create_pod(
    execution: Any,
    *,
    proof_run_id: str,
    repo_sha: str,
    attempt: int,
    controlled_bootstrap_failure: bool,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    """Create exactly one Secure Pod exposing two RTX 3090 CUDA devices."""
    shell = (
        base._controlled_bootstrap_failure_shell()
        if controlled_bootstrap_failure
        else launcher_v3.pod_shell_v3()
    )

    def create_once(gpu_ids: list[str]) -> dict[str, Any]:
        body: dict[str, object] = {
            "name": f"hephaestus-positive-promotion-{proof_run_id}-a{attempt}"[:180],
            "computeType": "GPU",
            "gpuCount": V3_GPU_COUNT,
            "gpuTypeIds": list(gpu_ids),
            "gpuTypePriority": "availability",
            "cloudType": "SECURE",
            "dataCenterIds": [base.launcher.DATACENTER_ID],
            "dataCenterPriority": "custom",
            "imageName": base.launcher.IMAGE,
            "containerDiskInGb": V3_CONTAINER_DISK_GB,
            "networkVolumeId": base.launcher.VOLUME_ID,
            "volumeMountPath": "/workspace",
            "dockerStartCmd": ["bash", "-lc", shell],
            "interruptible": False,
            "env": {
                "HEPHAESTUS_PROOF_RUN_ID": proof_run_id,
                "HEPHAESTUS_REPO_SHA": repo_sha,
                "HEPHAESTUS_ATTEMPT": str(attempt),
                "HEPHAESTUS_OPERATOR_APPROVAL_REF": base.launcher.APPROVAL_REF,
            },
        }
        return execution._create_pod(body)

    return base.launcher.create_with_capacity_retries(create_once)


# Reuse the already live-proven generic recovery implementation, while swapping
# only V3's scientific shell/verifier, capacity policy, and Pod topology.
base.launcher_v2 = launcher_v3
base.launcher.create_with_capacity_retries = _v3_create_with_capacity_retries
base._create_pod = _v3_create_pod


class RunPodPositivePromotionDriverV3(base.RunPodPositivePromotionDriver):
    def _write_launcher_record(self) -> None:
        base.launcher.atomic_json(
            Path("positive_promotion_launcher.json"),
            {
                "launcher_version": "positive-real-model-promotion-generic-loop.v3",
                "created_at": base._now(),
                "repo_sha": self.repo_sha,
                "proof_run_id": self.proof_run_id,
                "volume_id": base.launcher.VOLUME_ID,
                "datacenter_id": base.launcher.DATACENTER_ID,
                "generic_cli": "hephaestus run",
                "generic_recovery_owned": True,
                "scientific_variables_changed_on_retry": False,
                "proof_driver": "scripts/run_positive_promotion_proof_v3.py",
                "allowed_candidate_revisions": sorted(launcher_v3.ALLOWED_REVISIONS),
                "allowed_revision_licenses": dict(sorted(launcher_v3.ALLOWED_REVISION_LICENSES.items())),
                "gpu_count": V3_GPU_COUNT,
                "gpu_memory_floor_gb": V3_PER_GPU_MEMORY_GB,
                "aggregate_gpu_memory_floor_gb": V3_AGGREGATE_GPU_MEMORY_GB,
                "max_memory_gib_per_gpu": V3_MAX_MEMORY_GIB_PER_GPU,
                "model_parallelism": V3_MODEL_PARALLELISM,
                "gpu_type_ids": list(V3_GPU_IDS),
                "container_disk_gb": V3_CONTAINER_DISK_GB,
                "attempts": self.attempt_rows,
                "error": self.last_error,
                "status": "verified" if self.verification is not None else "running",
            },
        )

    def execute_cycle(self, *, runtime: Any, state: Any, cycle_index: int):
        result = super().execute_cycle(runtime=runtime, state=state, cycle_index=cycle_index)
        result.evidence["proof_driver"] = "scripts/run_positive_promotion_proof_v3.py"
        result.evidence["allowed_candidate_revisions"] = sorted(launcher_v3.ALLOWED_REVISIONS)
        result.evidence["allowed_revision_licenses"] = dict(sorted(launcher_v3.ALLOWED_REVISION_LICENSES.items()))
        result.evidence["gpu_count"] = V3_GPU_COUNT
        result.evidence["gpu_memory_floor_gb"] = V3_PER_GPU_MEMORY_GB
        result.evidence["aggregate_gpu_memory_floor_gb"] = V3_AGGREGATE_GPU_MEMORY_GB
        result.evidence["max_memory_gib_per_gpu"] = V3_MAX_MEMORY_GIB_PER_GPU
        result.evidence["model_parallelism"] = V3_MODEL_PARALLELISM
        result.evidence["gpu_type_ids"] = list(V3_GPU_IDS)
        result.evidence["container_disk_gb"] = V3_CONTAINER_DISK_GB
        return result


def build_driver(config: dict[str, object]) -> RunPodPositivePromotionDriverV3:
    return RunPodPositivePromotionDriverV3(
        prove_bootstrap_recovery_once=bool(config.get("prove_bootstrap_recovery_once", False)),
    )
