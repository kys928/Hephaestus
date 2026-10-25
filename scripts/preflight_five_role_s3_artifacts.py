#!/usr/bin/env python3
"""Read-only SHA/byte custody check of all five frozen S3 role adapters.

This is a pre-execution gate; it never calls model inference, loads trainable
weights into memory, changes a model, certifies scientific results or creates a
Pod. It uses only HEAD/GET on the configured existing RunPod network volume.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from hephaestus.providers.models.role_stack import (
    load_certified_role_model_stack, REQUIRED_ROLES,
)
from hephaestus.llm.role_contract import RoleMasteryContract

OUT = Path(os.environ.get("HEPHAESTUS_ROLE_ASSET_REPORT",
                          "evidence/cuda_role_assets.json"))


def main() -> int:
    import boto3
    from botocore.config import Config
    stack = load_certified_role_model_stack(
        "configs/models/hephaestus_role_model_stack_v1.json")
    contract = RoleMasteryContract.load(
        "configs/models/hephaestus_role_mastery_runtime_v1.json")
    report = {
        "schema": "hephaestus-frozen-five-role-assets-v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "identity_contract": "frozen_production_unchanged",
        "production_activation": False, "independent_certification": False,
        "role_assets": [], "complete": False, "issues": [],
    }
    for role in REQUIRED_ROLES:
        contract.validate_model(stack.resolve(role))
    names = ("RUNPOD_S3_ENDPOINT_URL", "RUNPOD_NETWORK_VOLUME_ID",
             "RUNPOD_DATACENTER_ID", "RUNPOD_S3_ACCESS_KEY_ID",
             "RUNPOD_S3_SECRET_ACCESS_KEY")
    if any(not os.environ.get(name) for name in names):
        report["issues"].append("read_only_volume_credentials_or_location_missing")
    else:
        client = boto3.client(
            "s3", endpoint_url=os.environ["RUNPOD_S3_ENDPOINT_URL"],
            region_name=os.environ["RUNPOD_DATACENTER_ID"],
            aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"],
            config=Config(retries={"mode": "standard", "max_attempts": 4},
                          connect_timeout=10, read_timeout=40))
        bucket = os.environ["RUNPOD_NETWORK_VOLUME_ID"]
        for role in REQUIRED_ROLES:
            spec = stack.resolve(role)
            item = {"role": role, "model_id": spec.model_id,
                    "revision": spec.revision,
                    "expected_bytes": spec.adapter.bytes,
                    "expected_sha256": spec.adapter.sha256,
                    "verified": False}
            try:
                head = client.head_object(Bucket=bucket, Key=spec.adapter.s3_key)
                length = int(head.get("ContentLength", -1))
                item["observed_bytes"] = length
                if length != spec.adapter.bytes:
                    raise ValueError("stored role adapter length mismatch")
                stream = client.get_object(Bucket=bucket, Key=spec.adapter.s3_key)["Body"]
                digest, total = hashlib.sha256(), 0
                try:
                    for data in iter(lambda: stream.read(1_048_576), b""):
                        total += len(data)
                        if total > spec.adapter.bytes:
                            raise ValueError("S3 adapter object exceeds pinned byte limit")
                        digest.update(data)
                finally:
                    stream.close()
                item["verified"] = (
                    total == spec.adapter.bytes
                    and digest.hexdigest() == spec.adapter.sha256)
                item["observed_sha256"] = digest.hexdigest()
                if not item["verified"]:
                    raise ValueError("role adapter original bytes changed")
            except Exception as exc:
                item["error_type"] = type(exc).__name__
                report["issues"].append(role + "_role_adapter_integrity_unverified")
            report["role_assets"].append(item)
    report["complete"] = (
        len(report["role_assets"]) == len(REQUIRED_ROLES)
        and all(item["verified"] for item in report["role_assets"])
        and not report["issues"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "schema": report["schema"], "complete": report["complete"],
        "roles": [{k: item[k] for k in ("role", "verified", "observed_bytes")
                   if k in item} for item in report["role_assets"]],
        "issues": report["issues"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
