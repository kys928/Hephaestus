#!/usr/bin/env python3
"""Launch and verify one frozen distance-to-role V1 shard on a single >=80GB RunPod GPU."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import launch_adaptation_elasticity_v1 as elastic_launch
import launch_positive_promotion_proof as launcher
import launch_positive_promotion_proof_v5 as v5
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig, RunPodExecutionAdapter

PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "configs/experiments/hephaestus_distance_to_role_v1.json"
TOPOLOGY_PATH = Path(__file__).resolve().parents[1] / "configs/eval_packs/hephaestus_cognitive_topology_v1.json"
CONTAINER_DISK_GB = 400
MAX_SECONDS = 19000
POLL_SECONDS = 15


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def run_id() -> str:
    explicit = os.environ.get("HEPHAESTUS_DTR_RUN_ID", "").strip()
    if explicit:
        return explicit
    return f"distance-to-role-v1-{required('GITHUB_RUN_ID')}"


def next_attempt(client: Any, execution_id: str) -> int:
    prefix = f"{launcher.SCIENTIFIC_PREFIX}/executions/{execution_id}/attempt-"
    attempts: set[int] = set()
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=launcher.VOLUME_ID, Prefix=prefix):
        for item in page.get("Contents", []):
            suffix = str(item.get("Key", ""))[len(prefix):]
            head = suffix.split("/", 1)[0]
            try:
                attempts.add(int(head))
            except ValueError:
                pass
    return max(attempts, default=0) + 1


def pod_shell() -> str:
    shell = v5.pod_shell_v5()
    shell = shell.replace(
        "/workspace/hephaestus/scientific/v1/model_admission/$HEPHAESTUS_REPO_SHA",
        "/workspace/hephaestus/scientific/v1/distance_to_role/model_admission/$HEPHAESTUS_REPO_SHA"
    )
    shell = shell.replace('"$PY" scripts/run_positive_promotion_proof_v5.py', '"$PY" scripts/run_distance_to_role_v1.py')
    return shell


def verify_shard(result: dict[str, Any], protocol: dict[str, Any], shard: str, expected_run_id: str) -> dict[str, Any]:
    if result.get("status") != "completed" or result.get("disposition") != "scientific_distance_shard_complete":
        raise RuntimeError(f"distance shard did not complete: {result.get('status')}: {result.get('disposition')}")
    if result.get("run_id") != expected_run_id or result.get("shard") != shard:
        raise RuntimeError("distance shard run/shard identity mismatch")
    protocol_sha = sha(PROTOCOL_PATH.read_bytes())
    topology_sha = sha(TOPOLOGY_PATH.read_bytes())
    if result.get("protocol_id") != protocol["protocol_id"] or result.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("distance shard protocol mismatch")
    if result.get("topology_protocol_sha256") != topology_sha:
        raise RuntimeError("distance shard topology hash mismatch")
    expected_models = list(protocol["execution"]["model_shards"][shard])
    models = result.get("models")
    if not isinstance(models, list) or [row.get("model_id") for row in models] != expected_models:
        raise RuntimeError("distance shard model coverage/order mismatch")
    expected_roles = set(protocol["roles"])
    expected_steps = [0, *[int(v) for v in protocol["training"]["dose_optimizer_steps"]]]
    for model in models:
        if model.get("status") != "complete" or set(model.get("roles", {})) != expected_roles:
            raise RuntimeError("distance shard role coverage incomplete")
        for role, curve in model["roles"].items():
            if curve.get("status") != "complete" or curve.get("role") != role:
                raise RuntimeError("distance role curve incomplete")
            if [int(row.get("optimizer_steps", -1)) for row in curve.get("points", [])] != expected_steps:
                raise RuntimeError("distance role curve dose coverage incomplete")
            if set(curve.get("threshold_distances", {})) != {"raw", "strict_safe", "bounded_safe"}:
                raise RuntimeError("distance role curve lacks threshold distances")
    if result.get("training_performed") is not True or result.get("promotion_performed") is not False or result.get("lineage_mutated") is not False:
        raise RuntimeError("distance shard governance invariants violated")
    return {
        "verification_version": "distance-to-role-launcher-verification.v1",
        "verified_at": now(), "run_id": expected_run_id, "shard": shard,
        "protocol_sha256": protocol_sha, "topology_protocol_sha256": topology_sha,
        "model_ids": expected_models, "role_count": len(expected_roles), "dose_steps": expected_steps,
        "promotion_performed": False, "lineage_mutated": False
    }


def maybe_existing(client: Any, protocol: dict[str, Any], shard: str, expected_run_id: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    key = f"{launcher.SCIENTIFIC_PREFIX}/distance_to_role/{expected_run_id}/shards/{shard}/shard_result.json"
    raw = launcher.base.maybe_read_key(client, key)
    if raw is None:
        return None
    payload = json.loads(raw)
    verification = verify_shard(payload, protocol, shard, expected_run_id)
    verification["source_key"] = key
    return payload, verification


def wait_result(client: Any, execution: RunPodExecutionAdapter, *, execution_id: str, attempt: int, pod_id: str) -> dict[str, Any]:
    key = f"{launcher.SCIENTIFIC_PREFIX}/executions/{execution_id}/attempt-{attempt}/driver_result.json"
    deadline = time.monotonic() + MAX_SECONDS
    while time.monotonic() < deadline:
        raw = launcher.base.maybe_read_key(client, key)
        if raw is not None:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise RuntimeError("distance terminal result is not an object")
            return payload
        launcher.base.pod_snapshot(execution, pod_id)
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"distance shard did not finish within {MAX_SECONDS} seconds")


def main() -> int:
    required("RUNPOD_API_KEY")
    repo_sha = required("GITHUB_SHA")
    shard = required("HEPHAESTUS_DISTANCE_SHARD")
    shared_run_id = run_id()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if shard not in protocol["execution"]["model_shards"]:
        raise RuntimeError(f"unknown distance shard: {shard}")
    execution_id = f"{shared_run_id}-{shard}"
    execution = RunPodExecutionAdapter(RunPodConfig.from_env(), EnvironmentSecretsProvider())
    client = launcher.base.s3_client()
    pod_id: str | None = None
    record: dict[str, Any] = {
        "launcher_version": "distance-to-role-runpod.v1", "started_at": now(), "run_id": shared_run_id,
        "execution_id": execution_id, "shard": shard, "repo_sha": repo_sha, "status": "starting",
        "gpu_type_ids": elastic_launch.GPU_IDS_80GB_PLUS, "gpu_count": 1, "container_disk_gb": CONTAINER_DISK_GB,
        "result_wait_timeout_seconds": MAX_SECONDS, "promotion_allowed": False, "lineage_mutation_allowed": False
    }
    try:
        client.head_bucket(Bucket=launcher.VOLUME_ID)
        existing = maybe_existing(client, protocol, shard, shared_run_id)
        if existing is not None:
            _, verification = existing
            launcher.atomic_json(Path(f"distance_to_role_verification_{shard}.json"), verification)
            record.update({"status": "verified_existing_terminal", "verification": verification, "teardown": {"not_created": True, "verified_absent": True}})
            print("DISTANCE_LAUNCH_JSON " + json.dumps({"run_id": shared_run_id, "shard": shard, "status": record["status"]}, sort_keys=True), flush=True)
            return 0

        pod_prefix = f"hephaestus-{execution_id}"
        record["prelaunch_cleanup"] = elastic_launch.cleanup_experiment_pods(execution, pod_prefix)
        attempt = next_attempt(client, execution_id)
        record["attempt"] = attempt

        def create_once() -> dict[str, Any]:
            return execution._create_pod({
                "name": pod_prefix[:180], "computeType": "GPU", "gpuCount": 1,
                "gpuTypeIds": elastic_launch.GPU_IDS_80GB_PLUS, "gpuTypePriority": "availability",
                "cloudType": "SECURE", "dataCenterIds": [launcher.DATACENTER_ID], "dataCenterPriority": "custom",
                "imageName": elastic_launch.routing.V4_IMAGE, "containerDiskInGb": CONTAINER_DISK_GB,
                "networkVolumeId": launcher.VOLUME_ID, "volumeMountPath": "/workspace",
                "dockerStartCmd": ["bash", "-lc", pod_shell()], "interruptible": False,
                "env": {
                    "HEPHAESTUS_PROOF_RUN_ID": execution_id,
                    "HEPHAESTUS_DTR_RUN_ID": shared_run_id,
                    "HEPHAESTUS_DISTANCE_SHARD": shard,
                    "HEPHAESTUS_REPO_SHA": repo_sha,
                    "HEPHAESTUS_ATTEMPT": str(attempt)
                }
            })

        observations: list[dict[str, Any]] = []
        pod = None
        for create_attempt in range(1, 13):
            try:
                pod = create_once()
                observations.append({"attempt": create_attempt, "status": "created", "pod_id": pod.get("id"), "gpu": pod.get("gpu")})
                break
            except Exception as exc:
                observations.append({"attempt": create_attempt, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                lowered = str(exc).lower()
                if any(term in lowered for term in ("402", "insufficient funds", "insufficient credit", "insufficient balance")) or create_attempt == 12:
                    raise
                time.sleep(10)
        if pod is None:
            raise RuntimeError("distance shard pod was not created")
        pod_id = str(pod["id"])
        record["pod_id"] = pod_id
        record["capacity_selection"] = observations
        terminal = wait_result(client, execution, execution_id=execution_id, attempt=attempt, pod_id=pod_id)
        verification = verify_shard(terminal, protocol, shard, shared_run_id)
        launcher.atomic_json(Path(f"distance_to_role_verification_{shard}.json"), verification)
        record.update({"status": "verified", "remote_status": terminal.get("status"), "verification": verification})
        print("DISTANCE_LAUNCH_JSON " + json.dumps({"run_id": shared_run_id, "shard": shard, "pod_id": pod_id, "status": "verified"}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if pod_id:
            record["teardown"] = elastic_launch.delete_pod_with_retries(execution, pod_id)
        else:
            record.setdefault("teardown", {"not_created": True, "verified_absent": True})
        record["completed_at"] = now()
        launcher.atomic_json(Path(f"distance_to_role_launcher_{shard}.json"), record)
        if pod_id and not record["teardown"].get("verified_absent"):
            raise RuntimeError(f"could not verify distance pod teardown: {pod_id}")


if __name__ == "__main__":
    raise SystemExit(main())
