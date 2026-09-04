#!/usr/bin/env python3
"""Run the first-scientific launcher with the current RunPod GPU enum."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from hephaestus.providers.runpod import RunPodExecutionAdapter


def _load() -> ModuleType:
    path = Path(__file__).with_name("launch_first_bounded_scientific_training.py")
    spec = importlib.util.spec_from_file_location("hephaestus_first_training_launcher", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load launcher: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    launcher = _load()

    def create_pod(execution: RunPodExecutionAdapter, run_id: str, repo_sha: str):
        return execution.create_bounded_gpu_pod(
            name=f"hephaestus-{run_id}"[:180],
            image_name=launcher.IMAGE,
            gpu_type_ids=[
                "NVIDIA GeForce RTX 3070",
                "NVIDIA GeForce RTX 3080",
                "NVIDIA GeForce RTX 3090",
                "NVIDIA L4",
                "NVIDIA GeForce RTX 4090",
            ],
            docker_start_cmd=["bash", "-lc", launcher.pod_shell()],
            env={
                "HEPHAESTUS_RUN_ID": run_id,
                "HEPHAESTUS_REPO_SHA": repo_sha,
                "HEPHAESTUS_OPERATOR_APPROVAL_REF": launcher.APPROVAL_REF,
                "HEPHAESTUS_MAX_WALL_SECONDS": "1200",
            },
            container_disk_in_gb=20,
            interruptible=False,
        )

    launcher.create_pod = create_pod
    return int(launcher.main())


if __name__ == "__main__":
    raise SystemExit(main())
