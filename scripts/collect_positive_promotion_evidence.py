#!/usr/bin/env python3
"""Collect decision-critical positive-promotion proof evidence from RunPod storage."""
from __future__ import annotations

import json
import os
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# One-shot forensic target: completed V4 proof whose cycle evidence showed zero
# persisted candidate samples for both model candidates.
PROOF_RUN_ID = "positive-real-model-promotion-001-34765826521"
REQUIRE_CYCLE_SUMMARY = True
BUCKET = os.environ["RUNPOD_NETWORK_VOLUME_ID"]
ENDPOINT = os.environ["RUNPOD_S3_ENDPOINT_URL"]
SCIENTIFIC_PREFIX = "hephaestus/scientific/v1"
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


def maybe_get(s3, key: str) -> bytes | None:
    try:
        return s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"NoSuchKey", "404", "NotFound"}:
            return None
        raise


def pull_prefix(s3, prefix: str, manifest: list[dict[str, object]]) -> None:
    token = None
    while True:
        kwargs = {"Bucket": BUCKET, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        response = s3.list_objects_v2(**kwargs)
        for item in response.get("Contents", []):
            key = str(item["Key"])
            raw = maybe_get(s3, key)
            if raw is None:
                continue
            relative = key.removeprefix(f"{SCIENTIFIC_PREFIX}/")
            path = OUT / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            manifest.append({"key": key, "bytes": len(raw)})
        if not response.get("IsTruncated"):
            return
        token = response.get("NextContinuationToken")


def main() -> int:
    s3 = client()
    OUT.mkdir(parents=True, exist_ok=True)
    proof_prefix = f"{SCIENTIFIC_PREFIX}/positive_promotion/{PROOF_RUN_ID}"
    exec_prefix = f"{SCIENTIFIC_PREFIX}/executions/{PROOF_RUN_ID}"

    manifest: list[dict[str, object]] = []
    # Pull complete decision cycles and, crucially, the evaluator-ready
    # generation reports/samples that live outside the cycle directories.
    for prefix in (
        f"{proof_prefix}/cycles/cycle-01/",
        f"{proof_prefix}/cycles/cycle-02/",
        f"{proof_prefix}/evaluations/",
        f"{exec_prefix}/attempt-1/",
    ):
        pull_prefix(s3, prefix, manifest)

    has_cycle_summary = any(str(item["key"]).endswith("cycle_summary.json") for item in manifest)
    generation_reports = [item for item in manifest if str(item["key"]).endswith("generation_report.json")]
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "proof_run_id": PROOF_RUN_ID,
        "files": len(manifest),
        "has_cycle_summary": has_cycle_summary,
        "generation_reports": len(generation_reports),
    }, sort_keys=True))
    if REQUIRE_CYCLE_SUMMARY and not has_cycle_summary:
        raise RuntimeError("no positive-promotion cycle summaries were found")
    if len(generation_reports) < 6:
        raise RuntimeError(f"expected at least six V4 candidate generation reports; found {len(generation_reports)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
