#!/usr/bin/env python3
"""Availability-priority wrapper for the second controlled training launcher."""
from __future__ import annotations

import json
from pathlib import Path

import launch_second_controlled_training as base
import launch_second_controlled_training_v2  # noqa: F401 - reviewed Pod runtime
from runpod_capacity_selection import create_with_capacity_retries

_capacity_observations: list[dict[str, object]] = []
_original_pod_shell = base.pod_shell


def _pod_shell_v3() -> str:
    shell = _original_pod_shell()
    shell = shell.replace(
        "scripts/run_second_controlled_training.py scripts/run_second_controlled_training_v2.py",
        "scripts/run_second_controlled_training.py scripts/run_second_controlled_training_v2.py scripts/run_second_controlled_training_v3.py",
    )
    shell = shell.replace(
        '"$PY" scripts/run_second_controlled_training_v2.py',
        '"$PY" scripts/run_second_controlled_training_v3.py',
    )
    if "run_second_controlled_training_v3.py" not in shell:
        raise RuntimeError("failed to project audited v3 controlled-training Pod command")
    return shell


def _create_capacity_aware(execution, repo_sha):
    global _capacity_observations

    def create_once(gpu_ids):
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


base.pod_shell = _pod_shell_v3
base.create_pod = _create_capacity_aware


if __name__ == "__main__":
    code = base.main()
    path = Path("second_controlled_training_launcher.json")
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["capacity_selection"] = _capacity_observations
        payload["training_driver"] = "scripts/run_second_controlled_training_v3.py"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raise SystemExit(code)
