#!/usr/bin/env python3
"""Collect adaptation-elasticity evidence from the S3-backed Network Volume and verify teardown."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import launch_positive_promotion_proof as launcher
import launch_adaptation_elasticity_v1 as elasticity_launcher
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig, RunPodExecutionAdapter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    run_id = os.environ.get("HEPHAESTUS_ELASTICITY_RUN_ID", "").strip() or elasticity_launcher.DEFAULT_RESUME_RUN_ID
    pod_prefix = f"hephaestus-{run_id}"
    out = Path("adaptation_elasticity_evidence")
    out.mkdir(exist_ok=True)
    s3 = launcher.base.s3_client()
    execution = RunPodExecutionAdapter(RunPodConfig.from_env(), EnvironmentSecretsProvider())

    inventory_observations: list[dict[str, object]] = []
    pods = elasticity_launcher.list_pods_with_retries(
        execution,
        observations=inventory_observations,
    )
    inventory = [{"id": pod.get("id"), "name": pod.get("name"), "desiredStatus": pod.get("desiredStatus"), "networkVolumeId": pod.get("networkVolumeId"), "gpu": pod.get("gpu", {}).get("displayName") if isinstance(pod.get("gpu"), dict) else None} for pod in pods]
    print("ELASTICITY_POD_INVENTORY_JSON " + json.dumps(inventory, sort_keys=True))
    if args.inventory_only:
        (out / "pods_before.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
        return 0

    cleanup = []
    for pod in inventory:
        if str(pod.get("name", "")).startswith(pod_prefix):
            cleanup.append(
                {
                    "pod_id": pod["id"],
                    **elasticity_launcher.delete_pod_with_retries(execution, pod["id"]),
                }
            )
    remaining = elasticity_launcher.list_pods_with_retries(
        execution,
        observations=inventory_observations,
    )
    dangling = [pod.get("id") for pod in remaining if str(pod.get("name", "")).startswith(pod_prefix)]
    cleanup_record = {"run_id": run_id, "cleanup": cleanup, "dangling_elasticity_pods": dangling, "remaining_pods": [{"id": pod.get("id"), "name": pod.get("name"), "desiredStatus": pod.get("desiredStatus")} for pod in remaining], "inventory_observations": inventory_observations}
    (out / "cleanup.json").write_text(json.dumps(cleanup_record, indent=2), encoding="utf-8")
    print("ELASTICITY_POD_CLEANUP_JSON " + json.dumps(cleanup_record, sort_keys=True))

    prefixes = [
        f"{launcher.SCIENTIFIC_PREFIX}/adaptation_elasticity/{run_id}/",
        f"{launcher.SCIENTIFIC_PREFIX}/executions/{run_id}/",
        f"{launcher.SCIENTIFIC_PREFIX}/adaptation_elasticity/model_admission/{os.environ['GITHUB_SHA']}/",
    ]
    manifest = []
    terminal_results: list[dict[str, object]] = []
    for prefix in prefixes:
        pages = elasticity_launcher.retry_transient(
            lambda selected_prefix=prefix: list(
                s3.get_paginator("list_objects_v2").paginate(
                    Bucket=launcher.VOLUME_ID,
                    Prefix=selected_prefix,
                )
            ),
            label=f"collect_list:{prefix}",
        )
        for page in pages:
            for item in page.get("Contents", []):
                key, size = item["Key"], int(item["Size"])
                if size > 32 * 1024 * 1024 or not key.endswith((".json", ".jsonl", ".txt", ".log")):
                    continue
                raw = elasticity_launcher.retry_transient(
                    lambda selected_key=key: launcher.base.read_key(s3, selected_key),
                    label=f"collect_read:{key}",
                )
                target = out / key
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                row = {"key": key, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                manifest.append(row)
                if key.endswith(("elasticity_result.json", "driver_result.json")):
                    parsed = json.loads(raw)
                    terminal_results.append({"key": key, "record": parsed})
                    print("ELASTICITY_RESULT_EVIDENCE_JSON " + json.dumps({**row, "record": parsed}, sort_keys=True))
                elif "/model_results/" in key and key.endswith(".json"):
                    record = json.loads(raw)
                    print("ELASTICITY_MODEL_RESULT_JSON " + json.dumps({**row, "model_id": record.get("model_id"), "mean_final_delta_quality_points": record.get("mean_final_delta_quality_points"), "mean_final_adapted_quality_100": record.get("mean_final_adapted_quality_100"), "mean_delta_per_gpu_hour": record.get("mean_delta_per_gpu_hour")}, sort_keys=True))
                elif key.endswith("pod_runtime.log"):
                    lines = raw.decode("utf-8", "replace").splitlines()
                    print("ELASTICITY_RUNTIME_TAIL " + json.dumps(lines[-100:]))

    protocol = json.loads(elasticity_launcher.PROTOCOL_PATH.read_text(encoding="utf-8"))
    completed = []
    for row in terminal_results:
        try:
            elasticity_launcher._verify(row["record"], protocol)
        except Exception:
            continue
        completed.append(row)
    record = {"run_id": run_id, "repo_sha": os.environ["GITHUB_SHA"], "github_workflow_run_id": os.environ["GITHUB_RUN_ID"], "files": manifest, "cleanup": cleanup_record, "validated_completed_results": [row["key"] for row in completed]}
    raw = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    (out / "collection_manifest.json").write_bytes(raw)
    key = f"{launcher.SCIENTIFIC_PREFIX}/adaptation_elasticity/{run_id}/collections/github-run-{os.environ['GITHUB_RUN_ID']}/collection_manifest.json"
    elasticity_launcher.retry_transient(
        lambda: s3.put_object(Bucket=launcher.VOLUME_ID, Key=key, Body=raw),
        label="write_collection_manifest",
    )
    readback = elasticity_launcher.retry_transient(
        lambda: launcher.base.read_key(s3, key),
        label="readback_collection_manifest",
    )
    if readback != raw:
        raise RuntimeError("elasticity collection manifest S3 readback mismatch")
    print("ELASTICITY_COLLECTION_MANIFEST_JSON " + json.dumps(record, sort_keys=True))
    if dangling:
        raise RuntimeError("temporary adaptation-elasticity Pods remain after cleanup")
    if not completed:
        raise RuntimeError("no validated scientific_elasticity_complete terminal result was collected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
