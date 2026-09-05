#!/usr/bin/env python3
from __future__ import annotations

import json
import os

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

PROOF_ID = "positive-real-model-promotion-001-33959818398"
ROOT = f"hephaestus/scientific/v1/positive_promotion/{PROOF_ID}"
EXEC = f"hephaestus/scientific/v1/executions/{PROOF_ID}"


def req(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise SystemExit(f"missing {name}")
    return v


def main() -> None:
    bucket = req("RUNPOD_NETWORK_VOLUME_ID")
    s3 = boto3.client(
        "s3",
        endpoint_url=req("RUNPOD_S3_ENDPOINT_URL").rstrip("/"),
        region_name=req("RUNPOD_DATACENTER_ID"),
        aws_access_key_id=req("RUNPOD_S3_ACCESS_KEY_ID"),
        aws_secret_access_key=req("RUNPOD_S3_SECRET_ACCESS_KEY"),
        config=Config(retries={"mode": "standard", "max_attempts": 5}),
    )

    keys = []
    for cycle in (1, 2):
        base = f"{ROOT}/cycles/cycle-{cycle:02d}"
        for name in (
            "cycle_summary.json",
            "comparison.json",
            "independent_review.json",
            "semantic_judge_exit.json",
            "promotion_gate.json",
            "production_loop_outcome.json",
        ):
            keys.append(f"{base}/{name}")
    keys += [
        f"{ROOT}/proof_result.json",
        f"{ROOT}/independent_verification.json",
    ]

    for key in keys:
        try:
            r = s3.get_object(Bucket=bucket, Key=key)
            raw = r["Body"].read()
            r["Body"].close()
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"NoSuchKey", "404"}:
                print(json.dumps({"key": key, "status": "missing"}))
                continue
            raise
        try:
            payload = json.loads(raw)
        except Exception:
            payload = raw.decode("utf-8", errors="replace")
        print(json.dumps({"key": key, "status": "found", "payload": payload}, sort_keys=True))

    for attempt in (1, 2, 3):
        key = f"{EXEC}/attempt-{attempt}/pod_runtime.log"
        try:
            r = s3.get_object(Bucket=bucket, Key=key)
            raw = r["Body"].read()
            r["Body"].close()
            text = raw.decode("utf-8", errors="replace")
            print(json.dumps({"key": key, "status": "found", "tail": text[-12000:]}))
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"NoSuchKey", "404"}:
                print(json.dumps({"key": key, "status": "missing"}))
            else:
                raise


if __name__ == "__main__":
    main()
