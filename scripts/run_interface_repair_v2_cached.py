#!/usr/bin/env python3
"""Interface Repair V2 entrypoint with persistent immutable parent caches."""
from __future__ import annotations

import hashlib
from typing import Any

import run_interface_repair_v2 as v2
from interface_repair_v2_bootstrap import load_stack_compatible, materialize_parent_cached


def put_bytes_verified_compatible(client: Any, key: str, raw: bytes, content_type: str) -> dict[str, Any]:
    """Verify RunPod S3 writes without assuming custom HEAD metadata survives.

    RunPod's S3-compatible endpoint may omit user metadata from HEAD responses.
    Keep the cheap length check, trust matching SHA metadata when present, and
    fall back to one GET+SHA readback only when the endpoint drops that metadata.
    """
    digest = hashlib.sha256(raw).hexdigest()
    bucket = v2.v1.bucket()
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=raw,
        ContentType=content_type,
        Metadata={"sha256": digest},
    )
    head = client.head_object(Bucket=bucket, Key=key)
    observed_length = int(head["ContentLength"])
    if observed_length != len(raw):
        raise RuntimeError(
            f"S3 size verification failed for {key}: {observed_length} != {len(raw)}"
        )

    observed_digest = str((head.get("Metadata") or {}).get("sha256", "")).strip()
    if observed_digest:
        if observed_digest != digest:
            raise RuntimeError(f"S3 metadata SHA-256 mismatch for {key}")
        return {"s3_key": key, "sha256": digest, "bytes": len(raw), "verification": "head_metadata"}

    response = client.get_object(Bucket=bucket, Key=key)
    try:
        observed = response["Body"].read()
    finally:
        response["Body"].close()
    if len(observed) != len(raw) or hashlib.sha256(observed).hexdigest() != digest:
        raise RuntimeError(f"S3 GET/SHA-256 verification failed for {key}")
    return {"s3_key": key, "sha256": digest, "bytes": len(raw), "verification": "get_sha256_fallback"}


# Execution-only substitutions. The frozen model revision, certified Phase-I
# source run, and adapter digest are unchanged. These adapters only normalize the
# registry schema, reuse persistent immutable caches, and accommodate the
# RunPod S3 endpoint's metadata behavior while preserving byte-level integrity.
v2.v1.load_stack = load_stack_compatible
v2.v1.materialize_parent = materialize_parent_cached
v2.put_bytes_verified = put_bytes_verified_compatible


if __name__ == "__main__":
    raise SystemExit(v2.main())
