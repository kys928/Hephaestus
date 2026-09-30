#!/usr/bin/env python3
from __future__ import annotations

import json
import os

import boto3
from botocore.config import Config

KEY = "hephaestus/scientific/v3/interface_repair_evaluator/interface-repair-v3-evaluator-preflight-36687021465/preflight/evaluator/samples/preflight_eval/shard-0001.jsonl"

client = boto3.client(
    "s3",
    endpoint_url=os.environ["RUNPOD_S3_ENDPOINT_URL"].rstrip("/"),
    region_name=os.environ["RUNPOD_DATACENTER_ID"],
    aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"],
    config=Config(retries={"mode": "standard", "max_attempts": 10}),
)
response = client.get_object(Bucket=os.environ["RUNPOD_NETWORK_VOLUME_ID"], Key=KEY)
try:
    body = response["Body"].read().decode("utf-8")
finally:
    response["Body"].close()

for line in body.splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    generation = row["generation"]
    score = row["score"]
    print(json.dumps({
        "case_id": row["case_id"],
        "generated_tokens": generation["generated_tokens"],
        "complete_contract_json": generation["complete_contract_json"],
        "contract_early_stop": generation["contract_early_stop"],
        "deadline_hit": generation["deadline_hit"],
        "extraction_reason": score.get("extraction_reason"),
        "raw_output": generation["raw_output"],
        "projected_output": generation["projected_output"],
    }, ensure_ascii=False, sort_keys=True))
