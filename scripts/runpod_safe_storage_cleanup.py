#!/usr/bin/env python3
"""Delete only explicitly approved disposable cache/temp objects from RunPod Network Volume.

Guardrails:
- never touches hephaestus/scientific/
- never touches persistent venv
- never deletes Hugging Face model blobs/snapshots
- only deletes approved temp/cache prefixes plus zero-byte HF lock/incomplete markers
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import boto3
from botocore.config import Config

APPROVED_PREFIXES = (
    "hephaestus-tmp/",
    "hephaestus/cache/xdg/pip/",
    "hephaestus-cache/xdg/",
    "hephaestus-cache/pip/",
)
HF_PREFIX = "hephaestus-cache/huggingface/"
FORBIDDEN_PREFIXES = (
    "hephaestus/scientific/",
    "hephaestus-cache/venv/",
)


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"missing environment variable: {name}")
    return value


def main() -> None:
    bucket = required("RUNPOD_NETWORK_VOLUME_ID")
    endpoint = required("RUNPOD_S3_ENDPOINT_URL").rstrip("/")
    region = required("RUNPOD_DATACENTER_ID")
    access_key = required("RUNPOD_S3_ACCESS_KEY_ID")
    secret_key = required("RUNPOD_S3_SECRET_ACCESS_KEY")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(retries={"mode": "standard", "max_attempts": 10}),
    )
    client.head_bucket(Bucket=bucket)
    paginator = client.get_paginator("list_objects_v2")

    candidates: dict[str, int] = {}
    reason: dict[str, str] = {}

    for prefix in APPROVED_PREFIXES:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, PaginationConfig={"PageSize": 1000}):
            for item in page.get("Contents", []) or []:
                key = str(item.get("Key", ""))
                if not key:
                    continue
                if key.startswith(FORBIDDEN_PREFIXES):
                    raise RuntimeError(f"guardrail violation: {key}")
                candidates[key] = int(item.get("Size", 0))
                reason[key] = f"approved_prefix:{prefix}"

    # Preserve all HF model data. Remove only zero-byte stale lock/incomplete markers.
    for page in paginator.paginate(Bucket=bucket, Prefix=HF_PREFIX, PaginationConfig={"PageSize": 1000}):
        for item in page.get("Contents", []) or []:
            key = str(item.get("Key", ""))
            size = int(item.get("Size", 0))
            if not key or size != 0:
                continue
            lower = key.lower()
            if lower.endswith(".lock") or lower.endswith(".incomplete"):
                candidates[key] = 0
                reason[key] = "zero_byte_hf_marker"

    keys = sorted(candidates)
    deleted = 0
    deleted_bytes = 0
    errors: list[dict[str, object]] = []
    for start in range(0, len(keys), 1000):
        batch = keys[start:start + 1000]
        response = client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": key} for key in batch], "Quiet": False},
        )
        deleted_keys = {str(row.get("Key")) for row in (response.get("Deleted") or [])}
        for key in deleted_keys:
            deleted += 1
            deleted_bytes += candidates.get(key, 0)
        for row in response.get("Errors") or []:
            errors.append(dict(row))

    # Verify every candidate is now absent. No broad delete verification is allowed.
    remaining: list[str] = []
    for prefix in APPROVED_PREFIXES:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, PaginationConfig={"PageSize": 1000}):
            for item in page.get("Contents", []) or []:
                key = str(item.get("Key", ""))
                if key in candidates:
                    remaining.append(key)
    for page in paginator.paginate(Bucket=bucket, Prefix=HF_PREFIX, PaginationConfig={"PageSize": 1000}):
        for item in page.get("Contents", []) or []:
            key = str(item.get("Key", ""))
            if key in candidates:
                remaining.append(key)

    report = {
        "cleanup_version": "runpod-safe-storage-cleanup.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "volume_id": bucket,
        "approved_prefixes": list(APPROVED_PREFIXES),
        "hf_policy": "preserve all HF data except zero-byte .lock/.incomplete markers",
        "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
        "candidate_objects": len(keys),
        "candidate_bytes": sum(candidates.values()),
        "deleted_objects": deleted,
        "deleted_bytes": deleted_bytes,
        "remaining_candidates": sorted(set(remaining)),
        "errors": errors,
        "success": not errors and not remaining,
    }
    with open("safe_storage_cleanup.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(json.dumps(report, sort_keys=True))
    if not report["success"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
