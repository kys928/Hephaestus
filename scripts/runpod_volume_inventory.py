#!/usr/bin/env python3
"""Fast read-only RunPod S3/network-volume storage audit.

Lists every object and aggregates byte usage by path prefix. It NEVER deletes or
modifies volume objects. Deletion candidates are advisory classifications only.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone

import boto3
from botocore.config import Config


def required(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise SystemExit(f"missing environment variable: {name}")
    return value


def classify(key: str) -> tuple[str, str]:
    lower = key.lower().lstrip("/")
    # High-confidence disposable runtime/cache material.
    if lower.startswith(("hephaestus-tmp/", "tmp/", "temp/")):
        return "disposable_temp", "high"
    if lower.startswith("hephaestus-cache/pip/"):
        return "reproducible_pip_cache", "high"
    if lower.startswith("hephaestus-cache/xdg/"):
        return "reproducible_xdg_cache", "high"
    if lower.startswith("hephaestus-cache/huggingface/") and (
        lower.endswith(".incomplete") or "/.locks/" in lower or lower.endswith(".lock")
    ):
        return "stale_or_incomplete_hf_cache", "high"
    # Reproducible but useful caches. Removing them causes re-download/rebuild.
    if lower.startswith("hephaestus-cache/huggingface/"):
        return "reproducible_hf_model_cache", "medium"
    if lower.startswith("hephaestus-cache/venv/"):
        return "reproducible_python_env", "medium"
    if lower.startswith("hephaestus-cache/interface-repair-v2/adapters/"):
        return "verified_adapter_cache", "low"
    # Scientific evidence and source artifacts are protected by default.
    if lower.startswith("hephaestus/scientific/"):
        return "scientific_evidence_keep", "protected"
    if lower.startswith("hephaestus/artifacts/") or lower.startswith("hephaestus/artifacts/"):
        return "scientific_artifact_keep", "protected"
    if lower.startswith("hephaestus/"):
        return "hephaestus_other_keep_review", "protected"
    if lower.endswith((".tmp", ".log", ".lock", ".pyc", ".pyo")):
        return "runtime_noise", "high"
    return "unclassified_review", "review"


def main() -> None:
    bucket = required("RUNPOD_NETWORK_VOLUME_ID")
    endpoint = required("RUNPOD_S3_ENDPOINT_URL").rstrip("/")
    region = required("RUNPOD_DATACENTER_ID")
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=required("RUNPOD_S3_ACCESS_KEY_ID"),
        aws_secret_access_key=required("RUNPOD_S3_SECRET_ACCESS_KEY"),
        config=Config(retries={"mode": "standard", "max_attempts": 10}),
    )
    client.head_bucket(Bucket=bucket)

    objects: list[dict[str, object]] = []
    prefix_bytes: dict[str, int] = defaultdict(int)
    prefix_objects: dict[str, int] = defaultdict(int)
    class_bytes: dict[str, int] = defaultdict(int)
    class_objects: dict[str, int] = defaultdict(int)
    class_confidence: dict[str, str] = {}

    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, PaginationConfig={"PageSize": 1000}):
        for item in page.get("Contents", []) or []:
            key = str(item.get("Key", ""))
            if not key:
                continue
            size = int(item.get("Size", 0))
            category, confidence = classify(key)
            row = {
                "key": key,
                "size": size,
                "last_modified": item.get("LastModified").isoformat() if item.get("LastModified") else None,
                "etag": str(item.get("ETag", "")).strip('"') or None,
                "category": category,
                "deletion_confidence": confidence,
            }
            objects.append(row)
            class_bytes[category] += size
            class_objects[category] += 1
            class_confidence[category] = confidence
            parts = [p for p in key.split("/") if p]
            for depth in range(1, min(5, len(parts)) + 1):
                prefix = "/".join(parts[:depth]) + ("/" if depth < len(parts) else "")
                prefix_bytes[prefix] += size
                prefix_objects[prefix] += 1

    objects.sort(key=lambda r: (-int(r["size"]), str(r["key"])))
    prefix_summary = [
        {"prefix": p, "bytes": prefix_bytes[p], "objects": prefix_objects[p]}
        for p in sorted(prefix_bytes, key=lambda p: (-prefix_bytes[p], p))
    ]
    category_summary = [
        {
            "category": c,
            "confidence": class_confidence[c],
            "bytes": class_bytes[c],
            "objects": class_objects[c],
        }
        for c in sorted(class_bytes, key=lambda c: (-class_bytes[c], c))
    ]
    summary = {
        "inventory_version": "runpod-volume-storage-audit.v3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "volume_id": bucket,
        "datacenter_id": region,
        "object_count": len(objects),
        "total_bytes": sum(int(o["size"]) for o in objects),
        "category_summary": category_summary,
        "largest_prefixes": prefix_summary[:250],
        "largest_objects": objects[:250],
    }
    with open("volume_inventory.json", "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "objects": objects}, fh, indent=2, sort_keys=True)
    with open("full_object_index.json", "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "objects": objects}, fh, indent=2, sort_keys=True)
    with open("candidate_paths.txt", "w", encoding="utf-8") as fh:
        for row in objects:
            if row["deletion_confidence"] in {"high", "medium", "low"}:
                fh.write(f"{row['deletion_confidence']}\t{row['category']}\t{row['size']}\t{row['key']}\n")

    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
