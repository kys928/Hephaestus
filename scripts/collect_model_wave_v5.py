#!/usr/bin/env python3
"""Collect governed small wave evidence, and verify temporary Pod cleanup."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import launch_positive_promotion_proof as launcher
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig, RunPodExecutionAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    run_id = "model-selection-v5-" + os.environ["GITHUB_RUN_ID"]
    out = Path("model_wave_evidence")
    out.mkdir(exist_ok=True)
    s3 = launcher.base.s3_client()
    execution = RunPodExecutionAdapter(RunPodConfig.from_env(), EnvironmentSecretsProvider())
    status, pods = execution._request("GET", "pods")
    if status != 200 or not isinstance(pods, list):
        raise RuntimeError(f"cannot verify RunPod inventory: HTTP {status}, type {type(pods).__name__}")
    inventory = [{"id": p.get("id"), "name": p.get("name"),
                  "desiredStatus": p.get("desiredStatus"), "networkVolumeId": p.get("networkVolumeId"),
                  "gpu": p.get("gpu", {}).get("displayName") if isinstance(p.get("gpu"), dict) else None}
                 for p in pods]
    print("POD_INVENTORY_JSON " + json.dumps(inventory, sort_keys=True))
    if args.inventory_only:
        (out / "pods_before.json").write_text(json.dumps(inventory, indent=2))
        return 0
    cleanup = []
    for pod in inventory:
        if str(pod["name"]).startswith(f"hephaestus-{run_id}-a"):
            cleanup.append({"pod_id": pod["id"], **launcher.base.delete_pod(execution, pod["id"])})
    status, remaining = execution._request("GET", "pods")
    if status != 200 or not isinstance(remaining, list):
        raise RuntimeError("cannot verify final Pod inventory")
    dangling = [p["id"] for p in remaining if str(p.get("name", "")).startswith(f"hephaestus-{run_id}-a")]
    cleanup_record = {"run_id": run_id, "cleanup": cleanup, "dangling_wave_pods": dangling,
                      "remaining_pods": [{"id": p.get("id"), "name": p.get("name"),
                                          "desiredStatus": p.get("desiredStatus")} for p in remaining]}
    (out / "cleanup.json").write_text(json.dumps(cleanup_record, indent=2))
    print("POD_CLEANUP_JSON " + json.dumps(cleanup_record, sort_keys=True))
    manifest = []
    prefixes = [f"{launcher.SCIENTIFIC_PREFIX}/positive_promotion/{run_id}/",
                f"{launcher.SCIENTIFIC_PREFIX}/executions/{run_id}/",
                f"{launcher.SCIENTIFIC_PREFIX}/model_admission/{os.environ['GITHUB_SHA']}/"]
    for prefix in prefixes:
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=launcher.VOLUME_ID, Prefix=prefix):
            for item in page.get("Contents", []):
                key, size = item["Key"], int(item["Size"])
                if size > 16 * 1024 * 1024 or not key.endswith((".json", ".jsonl", ".txt", ".log", ".sqlite", ".sqlite3")):
                    continue
                raw = s3.get_object(Bucket=launcher.VOLUME_ID, Key=key)["Body"].read()
                target = out / key
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                row = {"key": key, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                manifest.append(row)
                if key.endswith(".json") and len(raw) < 2 * 1024 * 1024 and any(
                    token in key for token in ("/cycles/", "/runtime/", "generation_report.json", "driver_result.json", "wave_result.json")):
                    print("WAVE_EVIDENCE_JSON " + json.dumps({**row, "record": json.loads(raw)}, sort_keys=True))
                if key.endswith("pod_runtime.log"):
                    lines = raw.decode("utf-8", "replace").splitlines()
                    print("POD_RUNTIME_TAIL " + json.dumps(lines[-100:]))
    record = {"run_id": run_id, "repo_sha": os.environ["GITHUB_SHA"], "files": manifest,
              "cleanup": cleanup_record}
    raw = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    (out / "collection_manifest.json").write_bytes(raw)
    key = f"{launcher.SCIENTIFIC_PREFIX}/positive_promotion/{run_id}/collection_manifest.json"
    s3.put_object(Bucket=launcher.VOLUME_ID, Key=key, Body=raw)
    if s3.get_object(Bucket=launcher.VOLUME_ID, Key=key)["Body"].read() != raw:
        raise RuntimeError("collection manifest S3 readback mismatch")
    print("COLLECTION_MANIFEST_JSON " + json.dumps(record, sort_keys=True))
    if dangling:
        raise RuntimeError("temporary wave Pods remain after cleanup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
