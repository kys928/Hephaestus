#!/usr/bin/env python3
from __future__ import annotations

import json
import os

import boto3
from botocore.config import Config

PROOF_ID = "positive-real-model-promotion-001-33959818398"
PREFIXES = [
    f"hephaestus/scientific/v1/positive_promotion/{PROOF_ID}/",
    f"hephaestus/scientific/v1/executions/{PROOF_ID}/",
]


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"missing {name}")
    return value


def main() -> None:
    bucket = required("RUNPOD_NETWORK_VOLUME_ID")
    client = boto3.client(
        "s3",
        endpoint_url=required("RUNPOD_S3_ENDPOINT_URL").rstrip("/"),
        region_name=required("RUNPOD_DATACENTER_ID"),
        aws_access_key_id=required("RUNPOD_S3_ACCESS_KEY_ID"),
        aws_secret_access_key=required("RUNPOD_S3_SECRET_ACCESS_KEY"),
        config=Config(retries={"mode": "standard", "max_attempts": 8}),
    )
    client.head_bucket(Bucket=bucket)
    paginator = client.get_paginator("list_objects_v2")
    for prefix in PREFIXES:
        print(json.dumps({"event": "prefix", "prefix": prefix}))
        keys: list[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            keys.extend(str(item["Key"]) for item in page.get("Contents") or [])
        for key in sorted(keys):
            obj = client.get_object(Bucket=bucket, Key=key)
            raw = obj["Body"].read()
            obj["Body"].close()
            print(json.dumps({"event": "object", "key": key, "bytes": len(raw)}))
            if key.endswith(".json") and len(raw) <= 2_000_000:
                try:
                    payload = json.loads(raw)
                    print(json.dumps({"event": "json", "key": key, "payload": payload}, sort_keys=True))
                except Exception as exc:
                    print(json.dumps({"event": "json_error", "key": key, "error": str(exc)}))
            elif key.endswith("pod_runtime.log"):
                text = raw.decode("utf-8", errors="replace")
                print(json.dumps({"event": "log_tail", "key": key, "tail": text[-20000:]}))


if __name__ == "__main__":
    main()
