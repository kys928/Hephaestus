#!/usr/bin/env python3
from __future__ import annotations

import json
import os

import boto3
from botocore.config import Config

PROOF = "positive-real-model-promotion-001-33959818398"
RUN = f"{PROOF}-candidate-1-repeat-1"
BASE = (
    f"hephaestus/scientific/v1/positive_promotion/{PROOF}/evaluations/{RUN}/"
    "semantic_generation/generation-settings-2ba1dd1322d04d4925c36ac4/samples"
)
SAMPLES = {
    "instruction_triplet:11": "9a20a0ac445c0452e5c43ffd.json",
    "instruction_triplet:29": "431e7e04891ee96f6f9b097d.json",
    "instruction_triplet:47": "9d690ff11c5d180cdcb2024c.json",
    "observatory_continuation:11": "bcc4a00849f64be26e23d252.json",
    "observatory_continuation:29": "a6be7fc5b6e2b4840e5a5e48.json",
    "observatory_continuation:47": "3045ddc72a33dd6adf118228.json",
}


def req(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"missing {name}")
    return value


def main() -> None:
    s3 = boto3.client(
        "s3",
        endpoint_url=req("RUNPOD_S3_ENDPOINT_URL").rstrip("/"),
        region_name=req("RUNPOD_DATACENTER_ID"),
        aws_access_key_id=req("RUNPOD_S3_ACCESS_KEY_ID"),
        aws_secret_access_key=req("RUNPOD_S3_SECRET_ACCESS_KEY"),
        config=Config(retries={"mode": "standard", "max_attempts": 5}),
    )
    bucket = req("RUNPOD_NETWORK_VOLUME_ID")
    for label, filename in SAMPLES.items():
        key = f"{BASE}/{filename}"
        obj = s3.get_object(Bucket=bucket, Key=key)
        raw = obj["Body"].read()
        obj["Body"].close()
        payload = json.loads(raw)
        compact = {
            "label": label,
            "key": key,
            "prompt": payload.get("prompt"),
            "output": payload.get("output", payload.get("generated_text", payload.get("text"))),
            "stop_reason": payload.get("stop_reason"),
            "finish_reason": payload.get("finish_reason"),
            "generated_tokens": payload.get("generated_tokens"),
            "scores": payload.get("scores"),
            "checks": payload.get("checks"),
            "metadata": payload.get("metadata"),
            "full_payload": payload,
        }
        print(json.dumps(compact, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
