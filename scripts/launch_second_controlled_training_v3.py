#!/usr/bin/env python3
"""Capacity-aware wrapper for the second controlled training launcher."""
from __future__ import annotations

import json
from pathlib import Path

import launch_second_controlled_training as base
import launch_second_controlled_training_v2  # noqa: F401 - reviewed Pod runtime
from runpod_capacity_selection import create_with_capacity_retries

_capacity_observations: list[dict[str, object]] = []
_original_create = base.create_pod


def _create_capacity_aware(execution, repo_sha):
    global _capacity_observations

    def create_once(gpu_ids):
        # Reproduce the base call with only execution routing changed.
        return execution.create_bounded_gpu_pod(
            name=f"hephaestus-{base.RUN_ID}"[:180],
            image_name=base.IMAGE,
            gpu_type_ids=list(gpu_ids),
            docker_start_cmd=["bash", "-lc", base.pod_shell()],
            env={
                "HEPHAESTUS_RUN_ID": base.RUN_ID,
                "HEPHAESTUS_REPO_SHA": repo_sha,
                "HEPHAESTUS_OPERATOR_APPROVAL_REF": base.APPROVAL_REF,
                "HEPHAESTUS_MAX_WALL_SECONDS": "1200",
            },
            container_disk_in_gb=20,
            interruptible=False,
        )

    pod, observations = create_with_capacity_retries(create_once)
    _capacity_observations = observations
    return pod


base.create_pod = _create_capacity_aware


if __name__ == "__main__":
    code = base.main()
    path = Path("second_controlled_training_launcher.json")
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["capacity_selection"] = _capacity_observations
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raise SystemExit(code)
