#!/usr/bin/env python3
"""Generic RunPod production-loop driver wrapper for promotion wave V4."""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import launch_positive_promotion_proof_v4 as launcher_v4  # noqa: E402
import runpod_positive_promotion_driver as base  # noqa: E402

# The V4 scientific protocol is frozen. Execution routing is infrastructure-only.
# The former exact 2x RTX 3090 topology repeatedly had zero stock in EU-CZ-1,
# so V4 now asks RunPod's live scheduler for one currently available >=48GB GPU.
# One such GPU comfortably fits Granite 8B and OLMo 13B in FP16 without
# quantization or CPU/disk model offload. Blackwell RTX PRO 6000 is included
# first because it is currently stocked in EU-CZ-1; additional >=48GB types are
# safe infrastructure fallbacks if availability changes between checks.
V4_GPU_IDS = (
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA L40S",
    "NVIDIA L40",
    "NVIDIA A40",
    "NVIDIA RTX A6000",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA H100 PCIe",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 NVL",
    "NVIDIA H200",
)
V4_GPU_COUNT = 1
V4_PER_GPU_MEMORY_GB = 48
V4_AGGREGATE_GPU_MEMORY_GB = V4_PER_GPU_MEMORY_GB
V4_MAX_MEMORY_GIB_PER_GPU = 44
V4_MODEL_PARALLELISM = "transformers_device_map_balanced_fp16"
V4_CONTAINER_DISK_GB = 400
V4_IMAGE = "pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime"
V4_MIN_CUDA_VERSION = "13.0"


def _v4_create_with_capacity_retries(
    create_once,
    *,
    attempts: int = 12,
    delay_seconds: float = 10.0,
):
    """Acquire any live EU-CZ-1 GPU that satisfies V4's >=48GB FP16 envelope."""
    observations: list[dict[str, object]] = []
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        observation: dict[str, object] = {
            "attempt": attempt,
            "queried_at": datetime.now(timezone.utc).isoformat(),
            "selection_source": "runpod_live_scheduler_v4_single_gpu_48gb_plus",
            "datacenter_id": base.launcher.DATACENTER_ID,
            "cloud_type": "SECURE",
            "gpu_count": V4_GPU_COUNT,
            "gpu_type_priority": "availability",
            "minimum_gpu_memory_gb": V4_PER_GPU_MEMORY_GB,
            "max_memory_gib_per_gpu": V4_MAX_MEMORY_GIB_PER_GPU,
            "model_parallelism": V4_MODEL_PARALLELISM,
            "ordered_gpu_type_ids": list(V4_GPU_IDS),
            "image": V4_IMAGE,
            "min_cuda_version": V4_MIN_CUDA_VERSION,
        }
        observations.append(observation)
        try:
            pod = create_once(list(V4_GPU_IDS))
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
            if any(
                phrase in lowered
                for phrase in (
                    "402",
                    "insufficient funds",
                    "insufficient credit",
                    "insufficient balance",
                    "account balance is too low",
                )
            ):
                raise
        if attempt < attempts:
            time.sleep(delay_seconds)
    raise RuntimeError(
        f"RunPod V4 >=48GB availability-priority Pod creation exhausted retries: {last_error}"
    ) from last_error


def _v4_create_pod(
    execution: Any,
    *,
    proof_run_id: str,
    repo_sha: str,
    attempt: int,
    controlled_bootstrap_failure: bool,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    """Create one Secure Pod on any live >=48GB GPU in EU-CZ-1."""
    shell = (
        base._controlled_bootstrap_failure_shell()
        if controlled_bootstrap_failure
        else launcher_v4.pod_shell_v4()
    )

    def create_once(gpu_ids: list[str]) -> dict[str, Any]:
        body: dict[str, object] = {
            "name": f"hephaestus-positive-promotion-v4-{proof_run_id}-a{attempt}"[:180],
            "computeType": "GPU",
            "gpuCount": V4_GPU_COUNT,
            "gpuTypeIds": list(gpu_ids),
            "gpuTypePriority": "availability",
            "cloudType": "SECURE",
            "dataCenterIds": [base.launcher.DATACENTER_ID],
            "dataCenterPriority": "custom",
            "imageName": V4_IMAGE,
            "minCudaVersion": V4_MIN_CUDA_VERSION,
            "containerDiskInGb": V4_CONTAINER_DISK_GB,
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


# Reuse the proven production-loop recovery policy while swapping only V4's
# shell/verifier and infrastructure routing. Scientific inputs remain frozen.
base.launcher_v2 = launcher_v4
base.launcher.create_with_capacity_retries = _v4_create_with_capacity_retries
base._create_pod = _v4_create_pod


class RunPodPositivePromotionDriverV4(base.RunPodPositivePromotionDriver):
    def _write_launcher_record(self) -> None:
        base.launcher.atomic_json(
            Path("positive_promotion_v4_launcher.json"),
            {
                "launcher_version": "positive-real-model-promotion-generic-loop.v4",
                "created_at": base._now(),
                "repo_sha": self.repo_sha,
                "proof_run_id": self.proof_run_id,
                "volume_id": base.launcher.VOLUME_ID,
                "datacenter_id": base.launcher.DATACENTER_ID,
                "generic_cli": "hephaestus run",
                "generic_recovery_owned": True,
                "scientific_variables_changed_on_retry": False,
                "proof_driver": "scripts/run_positive_promotion_proof_v4.py",
                "allowed_candidate_revisions": sorted(launcher_v4.ALLOWED_REVISIONS),
                "allowed_revision_licenses": dict(sorted(launcher_v4.ALLOWED_REVISION_LICENSES.items())),
                "gpu_count": V4_GPU_COUNT,
                "gpu_memory_floor_gb": V4_PER_GPU_MEMORY_GB,
                "aggregate_gpu_memory_floor_gb": V4_AGGREGATE_GPU_MEMORY_GB,
                "max_memory_gib_per_gpu": V4_MAX_MEMORY_GIB_PER_GPU,
                "model_parallelism": V4_MODEL_PARALLELISM,
                "gpu_type_ids": list(V4_GPU_IDS),
                "container_image": V4_IMAGE,
                "min_cuda_version": V4_MIN_CUDA_VERSION,
                "container_disk_gb": V4_CONTAINER_DISK_GB,
                "attempts": self.attempt_rows,
                "error": self.last_error,
                "status": "verified" if self.verification is not None else "running",
            },
        )

    def _write_verification_record(self) -> None:
        if self.verification is None:
            return
        base.launcher.atomic_json(Path("positive_promotion_v4_verification.json"), self.verification)

    def execute_cycle(self, *, runtime: Any, state: Any, cycle_index: int):
        result = super().execute_cycle(runtime=runtime, state=state, cycle_index=cycle_index)
        result.evidence["proof_driver"] = "scripts/run_positive_promotion_proof_v4.py"
        result.evidence["allowed_candidate_revisions"] = sorted(launcher_v4.ALLOWED_REVISIONS)
        result.evidence["allowed_revision_licenses"] = dict(sorted(launcher_v4.ALLOWED_REVISION_LICENSES.items()))
        result.evidence["gpu_count"] = V4_GPU_COUNT
        result.evidence["gpu_memory_floor_gb"] = V4_PER_GPU_MEMORY_GB
        result.evidence["aggregate_gpu_memory_floor_gb"] = V4_AGGREGATE_GPU_MEMORY_GB
        result.evidence["max_memory_gib_per_gpu"] = V4_MAX_MEMORY_GIB_PER_GPU
        result.evidence["model_parallelism"] = V4_MODEL_PARALLELISM
        result.evidence["gpu_type_ids"] = list(V4_GPU_IDS)
        result.evidence["container_image"] = V4_IMAGE
        result.evidence["min_cuda_version"] = V4_MIN_CUDA_VERSION
        result.evidence["container_disk_gb"] = V4_CONTAINER_DISK_GB
        return result


def build_driver(config: dict[str, object]) -> RunPodPositivePromotionDriverV4:
    return RunPodPositivePromotionDriverV4(
        prove_bootstrap_recovery_once=bool(config.get("prove_bootstrap_recovery_once", False)),
    )
