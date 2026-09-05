#!/usr/bin/env python3
"""Collect immutable positive-promotion proof evidence from the Network Volume."""
from __future__ import annotations

import json
import os
from pathlib import Path

import boto3
from botocore.config import Config

PROOF_RUN_ID = os.environ.get(
    "HEPHAESTUS_PROOF_RUN_ID",
    "positive-real-model-promotion-001-33957215257",
)
BUCKET = os.environ["RUNPOD_NETWORK_VOLUME_ID"]
ENDPOINT = os.environ["RUNPOD_S3_ENDPOINT_URL"]
PREFIX = f"hephaestus/scientific/v1/positive_promotion/{PROOF_RUN_ID}/"
EXEC_PREFIX = f"hephaestus/scientific/v1/executions/{PROOF_RUN_ID}/"
OUT = Path("positive_promotion_preserved_evidence")


def client():
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"],
        region_name="EU-CZ-1",
        config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
    )


def main() -> int:
    s3 = client()
    OUT.mkdir(parents=True, exist_ok=True)
    selected = []
    for prefix in (PREFIX, EXEC_PREFIX):
        token = None
        while True:
            kwargs = {"Bucket": BUCKET, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            page = s3.list_objects_v2(**kwargs)
            for item in page.get("Contents", []):
                key = str(item["Key"])
                name = key.rsplit("/", 1)[-1]
                if key.startswith(PREFIX) and name.endswith(".json"):
                    selected.append(key)
                elif key.startswith(EXEC_PREFIX) and name in {"driver_result.json", "pod_runtime.log"}:
                    selected.append(key)
            if not page.get("IsTruncated"):
                break
            token = page.get("NextContinuationToken")
    manifest = []
    for key in sorted(set(selected)):
        raw = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        relative = key.removeprefix("hephaestus/scientific/v1/")
        path = OUT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        manifest.append({"key": key, "bytes": len(raw)})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"proof_run_id": PROOF_RUN_ID, "files": len(manifest)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
