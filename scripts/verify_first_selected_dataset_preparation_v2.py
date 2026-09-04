#!/usr/bin/env python3
"""Run the independent verifier with byte-authoritative RunPod checks."""
from __future__ import annotations

import verify_first_selected_dataset_preparation as base


def _verify_named_by_bytes(
    s3: object,
    bucket: str,
    label: str,
    record: object,
) -> dict[str, object]:
    if not isinstance(record, dict):
        raise RuntimeError(f"runtime materialization {label} is missing")
    key = str(record.get("key") or "")
    expected = str(record.get("sha256") or "").removeprefix("sha256:")
    expected_size = int(record.get("byte_size") or -1)
    if not key or len(expected) != 64 or expected_size < 0:
        raise RuntimeError(f"runtime materialization {label} has invalid identity evidence")
    observed, size = base.stream_hash(s3, bucket, key)  # type: ignore[arg-type]
    if observed != expected or size != expected_size:
        raise RuntimeError(f"runtime materialization {label} failed independent SHA-256/size verification")
    head = s3.head_object(Bucket=bucket, Key=key)  # type: ignore[attr-defined]
    metadata = head.get("Metadata")
    metadata_digest = metadata.get("sha256") if isinstance(metadata, dict) else None
    return {
        "key": key,
        "sha256": f"sha256:{observed}",
        "byte_size": size,
        "custom_metadata_sha256": metadata_digest,
        "verification_basis": "full_get_object_sha256_and_exact_size",
    }


base.verify_named = _verify_named_by_bytes


if __name__ == "__main__":
    base.main()
