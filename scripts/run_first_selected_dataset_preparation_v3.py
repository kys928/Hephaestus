#!/usr/bin/env python3
"""Run selected-dataset preparation with byte-authoritative RunPod verification.

RunPod's S3-compatible API supports HeadObject/GetObject and files below 500 MB,
but integration evidence must not depend on AWS-specific preservation of custom
user metadata. Full SHA-256 readback plus exact byte size is authoritative here;
metadata, when present, remains supplemental evidence.
"""
from __future__ import annotations

import run_first_selected_dataset_preparation as base
import run_first_selected_dataset_preparation_v2  # noqa: F401 - installs single-PUT writers


def _verify_key_by_bytes(
    self: base.RunPodS3ArtifactStore,
    key: str,
    digest: str,
    size: int,
) -> bool:
    head = self._head(key)
    if head is None or int(head.get("ContentLength", -1)) != size:
        return False
    observed_hash, observed_size = base.stream_s3_hash(self.client, self.bucket, key)
    metadata = head.get("Metadata")
    metadata_digest = metadata.get("sha256") if isinstance(metadata, dict) else None
    print(
        "runpod_readback "
        f"key={key} size={observed_size} sha256={observed_hash} "
        f"custom_metadata_sha256={metadata_digest or 'absent'}"
    )
    return observed_hash == digest and observed_size == size


base.RunPodS3ArtifactStore._verify_key = _verify_key_by_bytes  # type: ignore[method-assign]


if __name__ == "__main__":
    base.main()
