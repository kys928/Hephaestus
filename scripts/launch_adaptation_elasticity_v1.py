#!/usr/bin/env python3
"""Launch the governed adaptation-elasticity V1 experiment on one >=80GB RunPod GPU."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import launch_positive_promotion_proof as launcher
import launch_positive_promotion_proof_v5 as v5
import runpod_positive_promotion_driver_v4 as routing
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig, RunPodExecutionAdapter

PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "configs/experiments/hephaestus_adaptation_elasticity_v1.json"
TOPOLOGY_PATH = Path(__file__).resolve().parents[1] / "configs/eval_packs/hephaestus_cognitive_topology_v1.json"
CONTAINER_DISK_GB = 400
# The first uninterrupted attempt completed 27/40 dose cells in four hours.
# Five hours and fifteen minutes is a bounded recovery window with room for the
# remaining 13 cells, immutable model downloads, evidence re-hashing, and
# teardown/collection inside GitHub-hosted runners' six-hour job ceiling.
ELASTICITY_MAX_SECONDS = 18900
launcher.MAX_SECONDS = ELASTICITY_MAX_SECONDS
DEFAULT_RESUME_RUN_ID = "adaptation-elasticity-v1-34961824753"
GPU_IDS_80GB_PLUS = [
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA H100 PCIe",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 NVL",
    "NVIDIA H200",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def _next_attempt(client: Any, proof_run_id: str) -> int:
    prefix = f"{launcher.SCIENTIFIC_PREFIX}/executions/{proof_run_id}/attempt-"
    attempts: set[int] = set()
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=launcher.VOLUME_ID,
        Prefix=prefix,
    ):
        for item in page.get("Contents", []):
            suffix = str(item.get("Key", ""))[len(prefix) :]
            head = suffix.split("/", 1)[0]
            try:
                attempts.add(int(head))
            except ValueError:
                continue
    return max(attempts, default=0) + 1


def pod_shell_elasticity() -> str:
    shell = v5.pod_shell_v5()
    shell = shell.replace(
        "/workspace/hephaestus/scientific/v1/model_admission/$HEPHAESTUS_REPO_SHA",
        "/workspace/hephaestus/scientific/v1/adaptation_elasticity/model_admission/$HEPHAESTUS_REPO_SHA",
    )
    shell = shell.replace('"$PY" scripts/run_positive_promotion_proof_v5.py', '"$PY" scripts/run_adaptation_elasticity_v1.py')
    return shell


def _verify(result: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    from build_adaptation_elasticity_dataset_v1 import build_dataset

    if result.get("status") != "completed" or result.get("disposition") != "scientific_elasticity_complete":
        raise RuntimeError(f"remote elasticity experiment did not complete: {result.get('status')}: {result.get('disposition')}")
    if result.get("protocol_id") != protocol["protocol_id"]:
        raise RuntimeError("remote elasticity protocol id mismatch")
    protocol_sha = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    topology_raw = TOPOLOGY_PATH.read_bytes()
    topology_sha = hashlib.sha256(topology_raw).hexdigest()
    topology = json.loads(topology_raw)
    if result.get("protocol_sha256") != protocol_sha or result.get("topology_protocol_sha256") != topology_sha:
        raise RuntimeError("remote elasticity protocol hashes mismatch")
    dataset_raw = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
        for row in build_dataset()
    ).encode()
    dataset_sha = hashlib.sha256(dataset_raw).hexdigest()
    if result.get("dataset_sha256") != dataset_sha or result.get("contamination_status") != "passed":
        raise RuntimeError("remote elasticity dataset/contamination identity mismatch")
    models = result.get("model_results")
    if not isinstance(models, list) or len(models) != 4:
        raise RuntimeError("remote elasticity result lacks all four models")
    expected_models = [(row["model_id"], row["revision"]) for row in topology["candidates"]]
    observed_models = [(row.get("model_id"), row.get("revision")) for row in models]
    if observed_models != expected_models:
        raise RuntimeError("remote elasticity candidate identities or order mismatch")
    expected_roles = list(protocol["roles"])
    for model in models:
        if model.get("status") != "complete" or list(model.get("roles", {}).keys()) != expected_roles:
            raise RuntimeError("remote elasticity model/role evidence is incomplete")
        for role in expected_roles:
            doses = model["roles"][role].get("doses")
            if not isinstance(doses, list) or [row.get("dose_epoch") for row in doses] != protocol["training"]["dose_checkpoints"]:
                raise RuntimeError("remote elasticity dose evidence is incomplete")
    if result.get("training_performed") is not True or result.get("promotion_performed") is not False or result.get("lineage_mutated") is not False:
        raise RuntimeError("elasticity governance invariants were violated")
    return {
        "verification_version": "adaptation-elasticity-launcher-verification.v1",
        "verified_at": _now(),
        "protocol_id": result["protocol_id"],
        "protocol_sha256": result["protocol_sha256"],
        "topology_protocol_sha256": result["topology_protocol_sha256"],
        "dataset_sha256": result["dataset_sha256"],
        "contamination_status": result["contamination_status"],
        "baseline_run_id": result["baseline_run_id"],
        "model_count": len(models),
        "role_count": len(expected_roles),
        "role_rankings": result.get("role_rankings", {}),
        "training_performed": True,
        "promotion_performed": False,
        "lineage_mutated": False,
    }


def main() -> int:
    _required("RUNPOD_API_KEY")
    repo_sha = _required("GITHUB_SHA")
    github_run_id = _required("GITHUB_RUN_ID")
    proof_run_id = os.environ.get("HEPHAESTUS_ELASTICITY_RUN_ID", "").strip() or DEFAULT_RESUME_RUN_ID
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    execution = RunPodExecutionAdapter(RunPodConfig.from_env(), EnvironmentSecretsProvider())
    client = launcher.base.s3_client()
    attempt = _next_attempt(client, proof_run_id)
    pod_id: str | None = None
    record: dict[str, Any] = {
        "launcher_version": "adaptation-elasticity-runpod.v1",
        "started_at": _now(),
        "proof_run_id": proof_run_id,
        "github_workflow_run_id": github_run_id,
        "repo_sha": repo_sha,
        "attempt": attempt,
        "resume_mode": True,
        "resume_source_run_id": proof_run_id,
        "protocol_id": protocol["protocol_id"],
        "container_image": routing.V4_IMAGE,
        "container_disk_gb": CONTAINER_DISK_GB,
        "gpu_type_ids": GPU_IDS_80GB_PLUS,
        "gpu_count": 1,
        "gpu_memory_class": ">=80GB",
        "result_wait_timeout_seconds": ELASTICITY_MAX_SECONDS,
        "scientific_variables_changed_on_retry": False,
        "promotion_allowed": False,
        "lineage_mutation_allowed": False,
    }
    try:
        client.head_bucket(Bucket=launcher.VOLUME_ID)

        def create_once(gpu_ids: list[str]) -> dict[str, Any]:
            return execution._create_pod({
                "name": f"hephaestus-{proof_run_id}"[:180],
                "computeType": "GPU",
                "gpuCount": 1,
                "gpuTypeIds": gpu_ids,
                "gpuTypePriority": "availability",
                "cloudType": "SECURE",
                "dataCenterIds": [launcher.DATACENTER_ID],
                "dataCenterPriority": "custom",
                "imageName": routing.V4_IMAGE,
                "containerDiskInGb": CONTAINER_DISK_GB,
                "networkVolumeId": launcher.VOLUME_ID,
                "volumeMountPath": "/workspace",
                "dockerStartCmd": ["bash", "-lc", pod_shell_elasticity()],
                "interruptible": False,
                "env": {
                    "HEPHAESTUS_PROOF_RUN_ID": proof_run_id,
                    "HEPHAESTUS_REPO_SHA": repo_sha,
                    "HEPHAESTUS_ATTEMPT": str(attempt),
                    "HEPHAESTUS_OPERATOR_APPROVAL_REF": launcher.APPROVAL_REF,
                },
            })

        observations: list[dict[str, Any]] = []
        last_error: BaseException | None = None
        for create_attempt in range(1, 13):
            try:
                pod = create_once(GPU_IDS_80GB_PLUS)
                observations.append({"attempt": create_attempt, "status": "created", "pod_id": pod.get("id"), "gpu": pod.get("gpu")})
                break
            except BaseException as exc:
                last_error = exc
                observations.append({"attempt": create_attempt, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                lowered = str(exc).lower()
                if any(term in lowered for term in ("402", "insufficient funds", "insufficient credit", "insufficient balance")):
                    raise
                if create_attempt == 12:
                    raise RuntimeError(f"RunPod >=80GB capacity retries exhausted: {last_error}") from last_error
                import time
                time.sleep(10)
        else:
            raise RuntimeError("RunPod elasticity pod was not created")

        pod_id = str(pod["id"])
        record["pod_id"] = pod_id
        record["capacity_selection"] = observations
        result, wait_observations = launcher.wait_for_result(client, execution, proof_run_id=proof_run_id, attempt=attempt, pod_id=pod_id)
        record["observations"] = wait_observations
        record["remote_status"] = result.get("status")
        record["remote_disposition"] = result.get("disposition")
        verification = _verify(result, protocol)
        launcher.atomic_json(Path("adaptation_elasticity_verification.json"), verification)
        record["verification"] = verification
        record["status"] = "verified"
        print("ADAPTATION_ELASTICITY_LAUNCH_JSON " + json.dumps({"proof_run_id": proof_run_id, "pod_id": pod_id, "status": "verified", "role_rankings": verification["role_rankings"]}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if pod_id:
            record["teardown"] = launcher.base.delete_pod(execution, pod_id)
        else:
            record["teardown"] = {"deleted": False, "not_created": True}
        record["completed_at"] = _now()
        launcher.atomic_json(Path("adaptation_elasticity_launcher.json"), record)


if __name__ == "__main__":
    raise SystemExit(main())
