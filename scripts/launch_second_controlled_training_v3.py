#!/usr/bin/env python3
"""Availability-priority wrapper for the second controlled training launcher."""
from __future__ import annotations

import json
from pathlib import Path

import launch_second_controlled_training as base
import launch_second_controlled_training_v2  # noqa: F401 - reviewed Pod runtime
from runpod_capacity_selection import create_with_capacity_retries

_capacity_observations: list[dict[str, object]] = []


def _create_capacity_aware(execution, repo_sha):
    global _capacity_observations

    def create_once(gpu_ids):
        # Same bounded Pod contract as the core adapter, except the scheduler is
        # explicitly allowed to choose the first actually available GPU from the
        # current allowlist. Datacenter, volume, Secure Cloud, image and one-GPU
        # shape remain fixed.
        body = {
            "name": f"hephaestus-{base.RUN_ID}"[:180],
            "computeType": "GPU",
            "gpuCount": 1,
            "gpuTypeIds": list(gpu_ids),
            "gpuTypePriority": "availability",
            "cloudType": "SECURE",
            "dataCenterIds": [base.DATACENTER_ID],
            "dataCenterPriority": "custom",
            "imageName": base.IMAGE,
            "containerDiskInGb": 20,
            "networkVolumeId": base.VOLUME_ID,
            "volumeMountPath": "/workspace",
            "dockerStartCmd": ["bash", "-lc", base.pod_shell()],
            "interruptible": False,
            "env": {
                "HEPHAESTUS_RUN_ID": base.RUN_ID,
                "HEPHAESTUS_REPO_SHA": repo_sha,
                "HEPHAESTUS_OPERATOR_APPROVAL_REF": base.APPROVAL_REF,
                "HEPHAESTUS_MAX_WALL_SECONDS": "1200",
            },
        }
        return execution._create_pod(body)  # integration-only scheduler projection

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
