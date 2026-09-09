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

# The proof loader places one FP16 model wholly on one CUDA device. V3's ~14B
# candidates therefore require a >=48GB-class scheduler allowlist; 24GB cards
# that are valid for V1/V2 would turn model loading into an infrastructure OOM.
# This is execution routing only and does not change scientific variables.
V3_GPU_IDS = (
    "NVIDIA A40",
    "NVIDIA L40",
    "NVIDIA L40S",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA H100 PCIe",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 NVL",
    "NVIDIA H200",
)


# Capacity in the volume's datacenter is transient. Keep the scheduler request
# alive for ~10 minutes per infrastructure cycle instead of giving up after
# roughly one minute. The generic production loop still owns the outer recovery
# budget, so this changes only operational capacity acquisition, never scientific
# variables, candidate order, model precision, or the >=48 GB memory floor.
def _v3_create_with_capacity_retries(create_once, *, attempts: int = 60, delay_seconds: float = 10.0):
    observations: list[dict[str, object]] = []
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        observation: dict[str, object] = {
            "attempt": attempt,
            "selection_source": "v3_fp16_single_gpu_high_memory_allowlist",
            "minimum_gpu_memory_class_gb": 48,
            "datacenter_id": base.launcher.DATACENTER_ID,
            "cloud_type": "SECURE",
            "gpu_count": 1,
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
    raise RuntimeError(f"RunPod V3 high-memory Pod creation exhausted retries: {last_error}") from last_error


# Reuse the already live-proven generic recovery implementation, but swap only
# the scientific Pod shell/verifier and the operational GPU capacity allowlist.
base.launcher_v2 = launcher_v3
base.launcher.create_with_capacity_retries = _v3_create_with_capacity_retries


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
                "gpu_memory_floor_gb": 48,
                "gpu_type_ids": list(V3_GPU_IDS),
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
        result.evidence["gpu_memory_floor_gb"] = 48
        result.evidence["gpu_type_ids"] = list(V3_GPU_IDS)
        return result


def build_driver(config: dict[str, object]) -> RunPodPositivePromotionDriverV3:
    return RunPodPositivePromotionDriverV3(
        prove_bootstrap_recovery_once=bool(config.get("prove_bootstrap_recovery_once", False)),
    )
