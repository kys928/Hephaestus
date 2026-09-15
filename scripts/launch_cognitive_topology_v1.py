#!/usr/bin/env python3
"""Launch one non-mutating cognitive-topology cohort on the proven RunPod GPU route."""
from __future__ import annotations

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

SPEC_PATH = Path(__file__).resolve().parents[1] / "configs/eval_packs/hephaestus_cognitive_topology_v1.json"
CONTAINER_DISK_GB = 400


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def pod_shell_topology() -> str:
    shell = v5.pod_shell_v5()
    shell = shell.replace(
        "/workspace/hephaestus/scientific/v1/model_admission/$HEPHAESTUS_REPO_SHA",
        "/workspace/hephaestus/scientific/v1/cognitive_topology/model_admission/$HEPHAESTUS_REPO_SHA",
    )
    shell = shell.replace(
        '"$PY" scripts/run_positive_promotion_proof_v5.py',
        '"$PY" scripts/run_cognitive_topology_v1.py',
    )
    return shell


def _verify(result: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") != "completed" or result.get("disposition") != "scientific_cohort_complete":
        raise RuntimeError(f"remote topology cohort did not complete: {result.get('status')}: {result.get('disposition')}")
    if result.get("protocol_id") != spec["protocol_id"]:
        raise RuntimeError("remote topology protocol id mismatch")
    summaries = result.get("model_summaries")
    if not isinstance(summaries, list) or len(summaries) != len(spec["candidates"]):
        raise RuntimeError("remote topology cohort lacks all model summaries")
    observed = [(row.get("model_id"), row.get("revision"), row.get("status")) for row in summaries if isinstance(row, dict)]
    expected = [(row["model_id"], row["revision"], "complete") for row in spec["candidates"]]
    if observed != expected:
        raise RuntimeError(f"remote topology candidate identity/status mismatch: {observed}")
    expected_samples = len(spec["cases"]) * int(spec["generation"]["repetitions"])
    if any(int(row.get("sample_count", -1)) != expected_samples for row in summaries):
        raise RuntimeError("remote topology sample count is incomplete")
    if result.get("training_performed") is not False or result.get("promotion_performed") is not False or result.get("lineage_mutated") is not False:
        raise RuntimeError("non-mutating topology invariants were violated")
    return {
        "verification_version": "cognitive-topology-launcher-verification.v1",
        "verified_at": _now(), "protocol_id": result["protocol_id"],
        "protocol_sha256": result["protocol_sha256"], "candidate_count": len(summaries),
        "case_count": len(spec["cases"]), "samples_per_model": expected_samples,
        "candidate_identities": observed, "specialization": result.get("specialization", {}),
        "runtime_gpus": {row["model_id"]: row.get("runtime", {}).get("gpu") for row in summaries},
        "training_performed": False, "promotion_performed": False, "lineage_mutated": False
    }


def main() -> int:
    _required("RUNPOD_API_KEY")
    repo_sha = _required("GITHUB_SHA")
    github_run_id = _required("GITHUB_RUN_ID")
    proof_run_id = f"cognitive-topology-v1-{github_run_id}"
    attempt = 1
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    execution = RunPodExecutionAdapter(RunPodConfig.from_env(), EnvironmentSecretsProvider())
    client = launcher.base.s3_client()
    pod_id: str | None = None
    record: dict[str, Any] = {
        "launcher_version": "cognitive-topology-runpod.v1", "started_at": _now(),
        "proof_run_id": proof_run_id, "repo_sha": repo_sha, "attempt": attempt,
        "protocol_id": spec["protocol_id"], "candidate_order": [row["model_id"] for row in spec["candidates"]],
        "container_image": routing.V4_IMAGE, "container_disk_gb": CONTAINER_DISK_GB,
        "gpu_type_ids": list(routing.V4_GPU_IDS), "gpu_count": 1,
        "scientific_variables_changed_on_retry": False, "non_mutating": True
    }
    try:
        client.head_bucket(Bucket=launcher.VOLUME_ID)

        def create_once(gpu_ids: list[str]) -> dict[str, Any]:
            return execution._create_pod({
                "name": f"hephaestus-{proof_run_id}"[:180],
                "computeType": "GPU", "gpuCount": 1, "gpuTypeIds": gpu_ids,
                "gpuTypePriority": "availability", "cloudType": "SECURE",
                "dataCenterIds": [launcher.DATACENTER_ID], "dataCenterPriority": "custom",
                "imageName": routing.V4_IMAGE, "containerDiskInGb": CONTAINER_DISK_GB,
                "networkVolumeId": launcher.VOLUME_ID, "volumeMountPath": "/workspace",
                "dockerStartCmd": ["bash", "-lc", pod_shell_topology()], "interruptible": False,
                "env": {
                    "HEPHAESTUS_PROOF_RUN_ID": proof_run_id, "HEPHAESTUS_REPO_SHA": repo_sha,
                    "HEPHAESTUS_ATTEMPT": str(attempt),
                    "HEPHAESTUS_OPERATOR_APPROVAL_REF": launcher.APPROVAL_REF
                }
            })

        pod, capacity = routing._v4_create_with_capacity_retries(create_once)
        pod_id = str(pod["id"])
        record["pod_id"] = pod_id
        record["capacity_selection"] = capacity
        result, observations = launcher.wait_for_result(
            client, execution, proof_run_id=proof_run_id, attempt=attempt, pod_id=pod_id
        )
        record["observations"] = observations
        record["remote_status"] = result.get("status")
        record["remote_disposition"] = result.get("disposition")
        verification = _verify(result, spec)
        launcher.atomic_json(Path("cognitive_topology_verification.json"), verification)
        record["verification"] = verification
        record["status"] = "verified"
        print("COGNITIVE_TOPOLOGY_LAUNCH_JSON " + json.dumps({
            "proof_run_id": proof_run_id, "pod_id": pod_id, "status": "verified",
            "specialization": verification["specialization"]
        }, sort_keys=True), flush=True)
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
        launcher.atomic_json(Path("cognitive_topology_launcher.json"), record)


if __name__ == "__main__":
    raise SystemExit(main())
