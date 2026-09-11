#!/usr/bin/env python3
"""Generic RunPod production-loop driver wrapper for promotion wave V4."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import launch_positive_promotion_proof_v4 as launcher_v4  # noqa: E402
import runpod_positive_promotion_driver as base  # noqa: E402
import runpod_positive_promotion_driver_v3 as topology  # noqa: E402

# V4 deliberately preserves the exact execution topology proven in V3 so
# candidate-model identity is the only scientific variable changed.
V4_GPU_IDS = topology.V3_GPU_IDS
V4_GPU_COUNT = topology.V3_GPU_COUNT
V4_PER_GPU_MEMORY_GB = topology.V3_PER_GPU_MEMORY_GB
V4_AGGREGATE_GPU_MEMORY_GB = topology.V3_AGGREGATE_GPU_MEMORY_GB
V4_MAX_MEMORY_GIB_PER_GPU = topology.V3_MAX_MEMORY_GIB_PER_GPU
V4_MODEL_PARALLELISM = topology.V3_MODEL_PARALLELISM
V4_CONTAINER_DISK_GB = topology.V3_CONTAINER_DISK_GB


def _v4_create_pod(
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
            "imageName": base.launcher.IMAGE,
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


# Reuse the already proven recovery policy and dual-3090 acquisition behavior.
base.launcher_v2 = launcher_v4
base.launcher.create_with_capacity_retries = topology._v3_create_with_capacity_retries
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
        result.evidence["container_disk_gb"] = V4_CONTAINER_DISK_GB
        return result


def build_driver(config: dict[str, object]) -> RunPodPositivePromotionDriverV4:
    return RunPodPositivePromotionDriverV4(
        prove_bootstrap_recovery_once=bool(config.get("prove_bootstrap_recovery_once", False)),
    )
