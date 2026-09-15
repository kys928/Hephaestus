#!/usr/bin/env python3
"""Collect cognitive-topology evidence from the S3-backed Network Volume and verify teardown."""
# Launch marker only: scientific cohort inputs and scoring remain unchanged.
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import launch_positive_promotion_proof as launcher
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig, RunPodExecutionAdapter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    run_id = "cognitive-topology-v1-" + os.environ["GITHUB_RUN_ID"]
    pod_prefix = f"hephaestus-{run_id}"
    out = Path("cognitive_topology_evidence")
    out.mkdir(exist_ok=True)
    s3 = launcher.base.s3_client()
    execution = RunPodExecutionAdapter(RunPodConfig.from_env(), EnvironmentSecretsProvider())

    status, pods = execution._request("GET", "pods")
    if status != 200 or not isinstance(pods, list):
        raise RuntimeError(f"cannot verify RunPod inventory: HTTP {status}")
    inventory = [
        {"id": pod.get("id"), "name": pod.get("name"), "desiredStatus": pod.get("desiredStatus"),
         "networkVolumeId": pod.get("networkVolumeId"),
         "gpu": pod.get("gpu", {}).get("displayName") if isinstance(pod.get("gpu"), dict) else None}
        for pod in pods
    ]
    print("TOPOLOGY_POD_INVENTORY_JSON " + json.dumps(inventory, sort_keys=True))
    if args.inventory_only:
        (out / "pods_before.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
        return 0

    cleanup = []
    for pod in inventory:
        if str(pod.get("name", "")).startswith(pod_prefix):
            cleanup.append({"pod_id": pod["id"], **launcher.base.delete_pod(execution, pod["id"])})
    status, remaining = execution._request("GET", "pods")
    if status != 200 or not isinstance(remaining, list):
        raise RuntimeError("cannot verify final RunPod inventory")
    dangling = [pod.get("id") for pod in remaining if str(pod.get("name", "")).startswith(pod_prefix)]
    cleanup_record = {
        "run_id": run_id, "cleanup": cleanup, "dangling_topology_pods": dangling,
        "remaining_pods": [{"id": pod.get("id"), "name": pod.get("name"), "desiredStatus": pod.get("desiredStatus")} for pod in remaining]
    }
    (out / "cleanup.json").write_text(json.dumps(cleanup_record, indent=2), encoding="utf-8")
    print("TOPOLOGY_POD_CLEANUP_JSON " + json.dumps(cleanup_record, sort_keys=True))

    prefixes = [
        f"{launcher.SCIENTIFIC_PREFIX}/cognitive_topology/{run_id}/",
        f"{launcher.SCIENTIFIC_PREFIX}/executions/{run_id}/",
        f"{launcher.SCIENTIFIC_PREFIX}/cognitive_topology/model_admission/{os.environ['GITHUB_SHA']}/",
    ]
    manifest = []
    for prefix in prefixes:
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=launcher.VOLUME_ID, Prefix=prefix):
            for item in page.get("Contents", []):
                key, size = item["Key"], int(item["Size"])
                if size > 32 * 1024 * 1024 or not key.endswith((".json", ".jsonl", ".txt", ".log")):
                    continue
                raw = s3.get_object(Bucket=launcher.VOLUME_ID, Key=key)["Body"].read()
                target = out / key
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                row = {"key": key, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                manifest.append(row)
                if key.endswith(("cohort_result.json", "driver_result.json")):
                    print("TOPOLOGY_RESULT_EVIDENCE_JSON " + json.dumps({**row, "record": json.loads(raw)}, sort_keys=True))
                elif "/model_summaries/" in key and key.endswith(".json"):
                    record = json.loads(raw)
                    print("TOPOLOGY_MODEL_SUMMARY_JSON " + json.dumps({
                        **row, "model_id": record.get("model_id"), "overall_quality_100": record.get("overall_quality_100"),
                        "roles": {name: data.get("quality_100") for name, data in record.get("roles", {}).items()}
                    }, sort_keys=True))
                elif key.endswith("pod_runtime.log"):
                    lines = raw.decode("utf-8", "replace").splitlines()
                    print("TOPOLOGY_RUNTIME_TAIL " + json.dumps(lines[-80:]))

    record = {"run_id": run_id, "repo_sha": os.environ["GITHUB_SHA"], "files": manifest, "cleanup": cleanup_record}
    raw = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    (out / "collection_manifest.json").write_bytes(raw)
    key = f"{launcher.SCIENTIFIC_PREFIX}/cognitive_topology/{run_id}/collection_manifest.json"
    s3.put_object(Bucket=launcher.VOLUME_ID, Key=key, Body=raw)
    if s3.get_object(Bucket=launcher.VOLUME_ID, Key=key)["Body"].read() != raw:
        raise RuntimeError("topology collection manifest S3 readback mismatch")
    print("TOPOLOGY_COLLECTION_MANIFEST_JSON " + json.dumps(record, sort_keys=True))
    if dangling:
        raise RuntimeError("temporary cognitive-topology Pods remain after cleanup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
