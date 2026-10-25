#!/usr/bin/env python3
"""Read-only, bounded and redacted diagnostic for a completed CUDA Pod attempt."""
from __future__ import annotations
import json, os, re
from pathlib import Path

import boto3
from botocore.config import Config

RUN_ID = os.environ.get("HEPHAESTUS_DIAGNOSTIC_RUN_ID", "hephaestus-five-role-cuda-40189242460")
if not re.fullmatch(r"hephaestus-five-role-cuda-[0-9]{8,20}", RUN_ID):
    raise ValueError("read-only diagnostic run identity invalid")
KEY = f"hephaestus/research/cuda-roles-20261025/{RUN_ID}/pod_runtime.log"
DEST = Path("evidence/cuda_bootstrap_diagnostic.json")


def redact(value: str) -> str:
    value = re.sub(r"(?i)(token|authorization|api[_-]?key|password|secret|access[_-]?key)\s*[:=]\s*\S+", r"\1=[REDACTED]", value)
    value = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", value)
    return value[:600]


def main() -> int:
    client = boto3.client(
        "s3", endpoint_url=os.environ["RUNPOD_S3_ENDPOINT_URL"],
        region_name=os.environ["RUNPOD_DATACENTER_ID"],
        aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"],
        config=Config(retries={"max_attempts": 3, "mode":"standard"},
                      connect_timeout=10, read_timeout=30))
    bucket = os.environ["RUNPOD_NETWORK_VOLUME_ID"]
    body = client.get_object(Bucket=bucket, Key=KEY)["Body"]
    raw = body.read(300_000)
    body.close()
    lines = raw.decode("utf-8", errors="replace").splitlines()
    relevant = [redact(x) for x in lines if (
        re.search(r"(?i)error|failed|traceback|not found|no such|fatal|404|403|cuda|python|command", x)
    )]
    payload = {
        "schema":"cuda-bootstrap-diagnostic.v1",
        "pod_run_id":RUN_ID,
        "read_only":True, "pod_recreated":False,
        "log_byte_sampled":len(raw), "log_line_count":len(lines),
        "relevant_lines":relevant[-30:],
        "tail_lines":[redact(s) for s in lines[-12:]],
    }
    DEST.parent.mkdir(exist_ok=True)
    DEST.write_text(json.dumps(payload,sort_keys=True,indent=2)+"\n")
    print(json.dumps(payload,sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
