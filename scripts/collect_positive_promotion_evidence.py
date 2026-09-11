#!/usr/bin/env python3
"""Collect decision-critical positive-promotion proof evidence from RunPod storage.

By default a completed/rejected wave must contain at least one cycle summary.
Operational monitoring may set HEPHAESTUS_REQUIRE_CYCLE_SUMMARY=false to collect
whatever immutable/partially-written execution evidence is already visible
without changing or interrupting the running scientific program.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

PROOF_RUN_ID = os.environ.get(
    "HEPHAESTUS_PROOF_RUN_ID",
    "positive-real-model-promotion-001-33957215257",
)
REQUIRE_CYCLE_SUMMARY = os.environ.get("HEPHAESTUS_REQUIRE_CYCLE_SUMMARY", "true").strip().lower() not in {
    "0", "false", "no"
}
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


def main() -> int:
    s3 = client()
    OUT.mkdir(parents=True, exist_ok=True)
    proof_prefix = f"{SCIENTIFIC_PREFIX}/positive_promotion/{PROOF_RUN_ID}"
    exec_prefix = f"{SCIENTIFIC_PREFIX}/executions/{PROOF_RUN_ID}"

    keys = [
        f"{proof_prefix}/cycles/cycle-01/cycle_summary.json",
        f"{proof_prefix}/cycles/cycle-02/cycle_summary.json",
        f"{proof_prefix}/cycles/cycle-01/experiment_comparison.json",
        f"{proof_prefix}/cycles/cycle-02/experiment_comparison.json",
        f"{proof_prefix}/cycles/cycle-01/independent_review.json",
        f"{proof_prefix}/cycles/cycle-02/independent_review.json",
        f"{proof_prefix}/cycles/cycle-01/semantic_judge_exit.json",
        f"{proof_prefix}/cycles/cycle-02/semantic_judge_exit.json",
        f"{proof_prefix}/cycles/cycle-01/certification_decision.json",
        f"{proof_prefix}/cycles/cycle-02/certification_decision.json",
        f"{proof_prefix}/cycles/cycle-01/promotion_gate_report.json",
        f"{proof_prefix}/cycles/cycle-02/promotion_gate_report.json",
        f"{proof_prefix}/proof_result.json",
    ]
    for attempt in range(1, 6):
        keys.extend(
            [
                f"{exec_prefix}/attempt-{attempt}/driver_result.json",
                f"{exec_prefix}/attempt-{attempt}/pod_runtime.log",
            ]
        )

    manifest = []
    for key in keys:
        raw = maybe_get(s3, key)
        if raw is None:
            continue
        relative = key.removeprefix(f"{SCIENTIFIC_PREFIX}/")
        path = OUT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        manifest.append({"key": key, "bytes": len(raw)})

    has_cycle_summary = any(str(item["key"]).endswith("cycle_summary.json") for item in manifest)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "proof_run_id": PROOF_RUN_ID,
        "files": len(manifest),
        "has_cycle_summary": has_cycle_summary,
        "require_cycle_summary": REQUIRE_CYCLE_SUMMARY,
    }, sort_keys=True))
    if REQUIRE_CYCLE_SUMMARY and not has_cycle_summary:
        raise RuntimeError("no positive-promotion cycle summaries were found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
