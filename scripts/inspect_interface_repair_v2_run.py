#!/usr/bin/env python3
"""Read-only inspection of one Interface Repair V2 run on the RunPod S3 volume."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config

RUN_ID = (os.environ.get("HEPHAESTUS_INTERFACE_REPAIR_RUN_ID") or "").strip()
if not RUN_ID:
    raise SystemExit("HEPHAESTUS_INTERFACE_REPAIR_RUN_ID is required")

bucket = os.environ["RUNPOD_NETWORK_VOLUME_ID"]
endpoint = os.environ["RUNPOD_S3_ENDPOINT_URL"].rstrip("/")
region = os.environ["RUNPOD_DATACENTER_ID"]
client = boto3.client(
    "s3",
    endpoint_url=endpoint,
    region_name=region,
    aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"],
    config=Config(retries={"mode": "standard", "max_attempts": 10}),
)

prefix = f"hephaestus/scientific/v2/interface_repair/{RUN_ID}/"
roles = ["planner", "evaluator", "judge", "controller"]


def get_json(key: str) -> dict[str, Any] | None:
    try:
        r = client.get_object(Bucket=bucket, Key=key)
    except Exception as exc:  # noqa: BLE001
        data = getattr(exc, "response", {})
        code = str(data.get("Error", {}).get("Code", "")) if isinstance(data, dict) else ""
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    try:
        value = json.loads(r["Body"].read().decode("utf-8"))
    finally:
        r["Body"].close()
    return value if isinstance(value, dict) else None


def compact_result(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    cert = value.get("certification") if isinstance(value.get("certification"), dict) else {}
    adapter = value.get("adapter") if isinstance(value.get("adapter"), dict) else {}
    return {
        "status": value.get("status"),
        "role": value.get("role"),
        "repo_sha": value.get("repo_sha"),
        "completed_at_unix": value.get("completed_at_unix"),
        "error_type": value.get("error_type"),
        "error": value.get("error"),
        "certified": cert.get("certified"),
        "post_interface_quality_100": (value.get("post_interface") or {}).get("quality_100") if isinstance(value.get("post_interface"), dict) else None,
        "post_regression_quality_100": (value.get("post_regression") or {}).get("quality_100") if isinstance(value.get("post_regression"), dict) else None,
        "adapter_sha256": adapter.get("sha256"),
        "adapter_s3_key": adapter.get("s3_key"),
    }

out: dict[str, Any] = {"run_id": RUN_ID, "prefix": prefix, "roles": {}}
for role in roles:
    base = f"{prefix}roles/{role}"
    out["roles"][role] = {
        "result": compact_result(get_json(f"{base}/result.json")),
        "progress": get_json(f"{base}/progress.json"),
    }

for candidate in ("live/result.json", "result.json"):
    value = get_json(prefix + candidate)
    if value is not None:
        out[candidate] = value

# Small object index for recovery diagnostics, without downloading large shards/adapters.
objects: list[dict[str, Any]] = []
token = None
while True:
    kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
    if token:
        kwargs["ContinuationToken"] = token
    page = client.list_objects_v2(**kwargs)
    for obj in page.get("Contents", []):
        key = str(obj.get("Key", ""))
        objects.append({"key": key, "bytes": int(obj.get("Size", 0))})
    if not page.get("IsTruncated"):
        break
    token = page.get("NextContinuationToken")

out["object_count"] = len(objects)
out["object_bytes"] = sum(x["bytes"] for x in objects)
out["notable_objects"] = [x for x in objects if x["key"].endswith(("result.json", "progress.json", "selected-adapter.tar.gz"))]
Path("interface_repair_v2_recovery_inspect.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(out, sort_keys=True))
