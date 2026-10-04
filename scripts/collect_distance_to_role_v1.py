#!/usr/bin/env python3
"""Collect S3 evidence, finalize the distance map, and independently verify RunPod teardown."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import launch_adaptation_elasticity_v1 as elastic_launch
import launch_distance_to_role_v1 as dtr_launch
import launch_positive_promotion_proof as launcher
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig, RunPodExecutionAdapter

PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "configs/experiments/hephaestus_distance_to_role_v1.json"
OUT = Path("distance_to_role_evidence")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def run_id() -> str:
    explicit = os.environ.get("HEPHAESTUS_DTR_RUN_ID", "").strip()
    if explicit:
        return explicit
    workflow = os.environ.get("GITHUB_RUN_ID", "").strip()
    if not workflow:
        raise RuntimeError("HEPHAESTUS_DTR_RUN_ID or GITHUB_RUN_ID is required")
    return f"distance-to-role-v1-{workflow}"


def read_json(client: Any, key: str) -> dict[str, Any]:
    raw = client.get_object(Bucket=launcher.VOLUME_ID, Key=key)["Body"].read()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError(f"S3 object is not a JSON object: {key}")
    return payload


def safe_rel(key: str, prefix: str) -> Path:
    rel = key[len(prefix):].lstrip("/")
    path = Path(rel)
    if not rel or path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"unsafe S3 evidence path: {key}")
    return path


def compact_key(key: str) -> bool:
    if "/adapter_model.safetensors" in key or key.endswith("/README.md"):
        return False
    return key.endswith((".json", ".jsonl", ".txt", ".log"))


def verify_teardown(execution: RunPodExecutionAdapter, prefix: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observations: list[dict[str, Any]] = []
    pods = elastic_launch.list_pods_with_retries(execution, observations=observations)
    dangling = [
        {"id": pod.get("id"), "name": pod.get("name"), "desiredStatus": pod.get("desiredStatus")}
        for pod in pods if str(pod.get("name", "")).startswith(prefix)
    ]
    return dangling, observations


def finalize(client: Any, execution: RunPodExecutionAdapter, rid: str, protocol: dict[str, Any]) -> dict[str, Any]:
    shard_results: dict[str, dict[str, Any]] = {}
    all_models: list[dict[str, Any]] = []
    for shard in protocol["execution"]["model_shards"]:
        key = f"{launcher.SCIENTIFIC_PREFIX}/distance_to_role/{rid}/shards/{shard}/shard_result.json"
        payload = read_json(client, key)
        dtr_launch.verify_shard(payload, protocol, shard, rid)
        shard_results[shard] = payload
        all_models.extend(payload["models"])
    expected_ids = [model_id for shard in protocol["execution"]["model_shards"].values() for model_id in shard]
    if [row["model_id"] for row in all_models] != expected_ids:
        raise RuntimeError("distance finalization model coverage/order mismatch")

    role_map: dict[str, list[dict[str, Any]]] = {role: [] for role in protocol["roles"]}
    model_map: dict[str, Any] = {}
    for model in all_models:
        model_entry = {"model_id": model["model_id"], "revision": model["revision"], "roles": {}}
        for role in protocol["roles"]:
            curve = model["roles"][role]
            point_map = {int(row["optimizer_steps"]): row for row in curve["points"]}
            selected = curve["selections"]
            def peak(name: str) -> dict[str, Any] | None:
                step = selected.get(name)
                if step is None:
                    return None
                point = point_map[int(step)]
                return {"optimizer_steps": int(step), "quality_100": float(point["role"]["quality_100"]), "semantic_score_delta": float(point["safety"]["semantic_score_delta"])}
            entry = {
                "baseline_quality_100": float(point_map[0]["role"]["quality_100"]),
                "raw_peak": peak("raw_peak_steps"),
                "strict_safe_peak": peak("strict_safe_peak_steps"),
                "bounded_safe_peak": peak("bounded_safe_peak_steps"),
                "threshold_distances": curve["threshold_distances"],
                "full_seed_confirmations": curve["full_seed_confirmations"],
                "trajectory": [
                    {
                        "optimizer_steps": int(point["optimizer_steps"]), "approx_epochs": float(point["approx_epochs"]),
                        "quality_100": float(point["role"]["quality_100"]),
                        "schema_compliance": float(point["role"]["schema_compliance"]),
                        "hallucination_rate": float(point["role"]["hallucination_rate"]),
                        "semantic_mean_score": float(point["semantic"]["mean_score"]),
                        "semantic_score_delta": float(point["safety"]["semantic_score_delta"]),
                        "new_hard_failure_count": len(point["safety"]["new_hard_failures"]),
                        "strict_safe": bool(point["safety"]["strict_safe"]), "bounded_safe": bool(point["safety"]["bounded_safe"])
                    }
                    for point in curve["points"]
                ]
            }
            model_entry["roles"][role] = entry
            role_map[role].append({"model_id": model["model_id"], **entry})
        model_map[model["model_id"]] = model_entry

    dangling, observations = verify_teardown(execution, f"hephaestus-{rid}")
    if dangling:
        raise RuntimeError(f"distance finalizer found dangling pods: {dangling}")
    distance_map = {
        "result_version": "distance-to-role-map.v1",
        "status": "completed", "disposition": "scientific_distance_complete",
        "run_id": rid, "protocol_id": protocol["protocol_id"], "protocol_sha256": sha(PROTOCOL_PATH.read_bytes()),
        "training_dataset_sha256": shard_results[next(iter(shard_results))]["training_dataset_sha256"],
        "semantic_content_hash": protocol["sources"]["semantic_content_hash"],
        "thresholds_100": protocol["role_screening"]["quality_thresholds_100"],
        "model_count": len(all_models), "role_count": len(protocol["roles"]),
        "models": model_map, "roles": role_map,
        "shards": {name: {"model_ids": [row["model_id"] for row in payload["models"]], "status": payload["status"]} for name, payload in shard_results.items()},
        "teardown_verified": True, "dangling_pods": [], "inventory_observations": observations,
        "training_performed": True, "promotion_performed": False, "lineage_mutated": False
    }
    raw = (json.dumps(distance_map, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    key = f"{launcher.SCIENTIFIC_PREFIX}/distance_to_role/{rid}/distance_map.json"
    prior = launcher.base.maybe_read_key(client, key)
    if prior is not None and prior != raw:
        raise RuntimeError("existing distance map differs from deterministic finalization")
    if prior is None:
        client.put_object(Bucket=launcher.VOLUME_ID, Key=key, Body=raw)
    observed = client.get_object(Bucket=launcher.VOLUME_ID, Key=key)["Body"].read()
    if observed != raw:
        raise RuntimeError("distance map S3 readback mismatch")
    return distance_map


def collect(client: Any, rid: str) -> dict[str, Any]:
    prefixes = {
        "scientific": f"{launcher.SCIENTIFIC_PREFIX}/distance_to_role/{rid}/",
        "executions": f"{launcher.SCIENTIFIC_PREFIX}/executions/{rid}-"
    }
    OUT.mkdir(exist_ok=True)
    objects: list[dict[str, Any]] = []
    for namespace, prefix in prefixes.items():
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=launcher.VOLUME_ID, Prefix=prefix):
            for item in page.get("Contents", []):
                key = str(item["Key"])
                if not compact_key(key):
                    continue
                raw = client.get_object(Bucket=launcher.VOLUME_ID, Key=key)["Body"].read()
                target = OUT / namespace / safe_rel(key, prefix)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                objects.append({"namespace": namespace, "key": key, "bytes": len(raw), "sha256": sha(raw)})
    return {"object_count": len(objects), "objects": objects, "prefixes": prefixes}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", choices=("olmo-granite", "qwen-ministral"))
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    rid = run_id()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    client = launcher.base.s3_client()
    execution = RunPodExecutionAdapter(RunPodConfig.from_env(), EnvironmentSecretsProvider())

    final_result = None
    if args.finalize:
        final_result = finalize(client, execution, rid, protocol)
    collection = collect(client, rid)
    if args.shard:
        dangling, observations = verify_teardown(execution, f"hephaestus-{rid}-{args.shard}")
    else:
        dangling, observations = verify_teardown(execution, f"hephaestus-{rid}")
    manifest = {
        "collection_version": "distance-to-role-collector.v1", "run_id": rid, "shard": args.shard,
        **collection, "dangling_pods": dangling, "inventory_observations": observations,
        "finalized": final_result is not None,
        "distance_map_status": final_result.get("status") if final_result else None
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "collection_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if final_result is not None:
        (OUT / "distance_map.json").write_text(json.dumps(final_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("DISTANCE_COLLECTION_JSON " + json.dumps({"run_id": rid, "shard": args.shard, "object_count": collection["object_count"], "dangling_pods": dangling, "finalized": final_result is not None}, sort_keys=True))
    if dangling:
        raise RuntimeError(f"distance collector found dangling pods: {dangling}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
