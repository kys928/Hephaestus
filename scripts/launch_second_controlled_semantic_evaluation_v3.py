#!/usr/bin/env python3
"""Capacity-aware wrapper for the second controlled semantic-evaluation launcher."""
from __future__ import annotations

import json
from pathlib import Path

import launch_second_controlled_semantic_evaluation as base
import launch_second_controlled_semantic_evaluation_v2  # noqa: F401 - reviewed Pod runtime
from runpod_capacity_selection import create_with_capacity_retries

_capacity_observations: list[dict[str, object]] = []


def _create_capacity_aware(execution, eval_run_id, repo_sha, candidate_checkpoint_hash):
    global _capacity_observations

    def create_once(gpu_ids):
        return execution.create_bounded_gpu_pod(
            name=f"hephaestus-{eval_run_id}"[:180],
            image_name=base.IMAGE,
            gpu_type_ids=list(gpu_ids),
            docker_start_cmd=["bash", "-lc", base.pod_shell()],
            env={
                "HEPHAESTUS_EVAL_RUN_ID": eval_run_id,
                "HEPHAESTUS_REPO_SHA": repo_sha,
                "HEPHAESTUS_CANDIDATE_CHECKPOINT_HASH": candidate_checkpoint_hash,
                "HEPHAESTUS_OPERATOR_APPROVAL_REF": base.APPROVAL_REF,
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
    path = Path("second_controlled_semantic_evaluation_launcher.json")
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["capacity_selection"] = _capacity_observations
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raise SystemExit(code)
