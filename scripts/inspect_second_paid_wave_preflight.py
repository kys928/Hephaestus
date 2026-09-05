#!/usr/bin/env python3
"""Read-only S3 preflight for the second paid GPU wave."""
from __future__ import annotations

import json
import os
from typing import Any

import boto3
from botocore.config import Config


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"missing environment variable: {name}")
    return value


def main() -> None:
    bucket = required("RUNPOD_NETWORK_VOLUME_ID")
    client = boto3.client(
        "s3",
        endpoint_url=required("RUNPOD_S3_ENDPOINT_URL").rstrip("/"),
        region_name=required("RUNPOD_DATACENTER_ID"),
        aws_access_key_id=required("RUNPOD_S3_ACCESS_KEY_ID"),
        aws_secret_access_key=required("RUNPOD_S3_SECRET_ACCESS_KEY"),
        config=Config(retries={"mode": "standard", "max_attempts": 10}),
    )
    client.head_bucket(Bucket=bucket)

    eval_prefix = (
        "hephaestus/scientific/v1/evaluations/"
        "second-controlled-semantic-evaluation-001-33889959923/generation/"
        "planned-run-b8e558e54effac85/"
    )
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=eval_prefix):
        keys.extend(str(item["Key"]) for item in (page.get("Contents") or []))
    sample_keys = sorted(key for key in keys if "/samples/" in key and key.endswith(".json"))
    print(json.dumps({"event": "candidate_sample_inventory", "count": len(sample_keys), "prefix": eval_prefix}))
    if len(sample_keys) != 18:
        raise SystemExit(f"expected 18 candidate samples, found {len(sample_keys)}")

    for key in sample_keys:
        body = client.get_object(Bucket=bucket, Key=key)["Body"]
        raw = body.read()
        body.close()
        payload: Any = json.loads(raw)
        print(json.dumps({"event": "candidate_sample", "key": key, "payload": payload}, sort_keys=True))

    data_prefix = "hephaestus/scientific/v1/runtime_bindings/planned-run-b8e558e54effac85/dataset/"
    data_keys: list[dict[str, object]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=data_prefix):
        for item in page.get("Contents") or []:
            data_keys.append({"key": str(item["Key"]), "size": int(item.get("Size", 0))})
    print(json.dumps({"event": "dataset_binding_inventory", "objects": data_keys}, sort_keys=True))

    train_key = next((str(item["key"]) for item in data_keys if str(item["key"]).endswith("trainable.jsonl")), None)
    if train_key:
        response = client.get_object(Bucket=bucket, Key=train_key, Range="bytes=0-65535")
        preview = response["Body"].read().decode("utf-8", errors="replace")
        response["Body"].close()
        lines = preview.splitlines()[:8]
        print(json.dumps({"event": "trainable_head", "key": train_key, "lines": lines}, sort_keys=True))
    else:
        print(json.dumps({"event": "trainable_head", "status": "not_found_under_runtime_binding_prefix"}))


if __name__ == "__main__":
    main()
