#!/usr/bin/env python3
"""V5 adapter for the existing RunPod production loop and live GPU router."""
from __future__ import annotations

import os
from pathlib import Path
import runpod_positive_promotion_driver_v4 as routing
import launch_positive_promotion_proof_v5 as wave

base = routing.base
base.launcher_v2 = wave
base.launcher.MAX_SECONDS = 3000


def create_pod(execution, *, proof_run_id, repo_sha, attempt, controlled_bootstrap_failure):
    if controlled_bootstrap_failure:
        raise ValueError("controlled failures are disabled for the admitted scientific wave")

    def once(gpu_ids):
        return execution._create_pod({
            "name": f"hephaestus-{proof_run_id}-a{attempt}",
            "computeType": "GPU", "gpuCount": 1, "gpuTypeIds": gpu_ids,
            "gpuTypePriority": "availability", "cloudType": "SECURE",
            "dataCenterIds": [base.launcher.DATACENTER_ID], "dataCenterPriority": "custom",
            "imageName": routing.V4_IMAGE, "containerDiskInGb": routing.V4_CONTAINER_DISK_GB,
            "networkVolumeId": base.launcher.VOLUME_ID, "volumeMountPath": "/workspace",
            "dockerStartCmd": ["bash", "-lc", wave.pod_shell_v5()], "interruptible": False,
            "env": {"HEPHAESTUS_PROOF_RUN_ID": proof_run_id, "HEPHAESTUS_REPO_SHA": repo_sha,
                    "HEPHAESTUS_ATTEMPT": str(attempt),
                    "HEPHAESTUS_OPERATOR_APPROVAL_REF": base.launcher.APPROVAL_REF},
        })
    return routing._v4_create_with_capacity_retries(once)


base._create_pod = create_pod


class RunPodModelWaveDriver(base.RunPodPositivePromotionDriver):
    def _write_launcher_record(self):
        base.launcher.atomic_json(Path("model_selection_v5_launcher.json"), {
            "launcher_version": "admitted-model-wave.v5", "created_at": base._now(),
            "repo_sha": self.repo_sha, "proof_run_id": self.proof_run_id,
            "volume_id": base.launcher.VOLUME_ID, "datacenter_id": base.launcher.DATACENTER_ID,
            "container_image": routing.V4_IMAGE, "container_disk_gb": routing.V4_CONTAINER_DISK_GB,
            "gpu_type_ids": list(routing.V4_GPU_IDS), "gpu_count": 1, "max_memory_gib": 44,
            "proof_driver": "scripts/run_positive_promotion_proof_v5.py",
            "allowed_revision_licenses": wave.ALLOWED_REVISION_LICENSES,
            "scientific_variables_changed_on_retry": False,
            "attempts": self.attempt_rows, "error": self.last_error,
            "status": "verified" if self.verification is not None else "failed" if self.last_error else "running",
        })


def build_driver(config):
    return RunPodModelWaveDriver(
        prove_bootstrap_recovery_once=False,
        proof_run_id="model-selection-v5-" + os.environ["GITHUB_RUN_ID"],
    )
