#!/usr/bin/env python3
"""Run the first scientific bootstrap with RunPod-compatible object verification.

RunPod Network Volumes expose an S3-compatible filesystem layer. Custom user
metadata is not part of Hephaestus' immutable identity contract, so verification
uses the portable operations RunPod documents: HeadObject for exact byte size and
GetObject for a full SHA-256 readback. The content-addressed key remains the
immutable identity.
"""

from __future__ import annotations

import hashlib

from scripts import bootstrap_first_scientific_experiment as bootstrap


def _verify_key(self, key: str, digest: str, byte_size: int) -> bool:
    head = self._head(key)
    if head is None:
        return False
    if int(head.get("ContentLength", -1)) != byte_size:
        raise RuntimeError(f"immutable RunPod object has unexpected byte size: {key}")

    response = self.client.get_object(Bucket=self.bucket, Key=key)
    body = response["Body"]
    hasher = hashlib.sha256()
    observed = 0
    try:
        while True:
            chunk = body.read(1024 * 1024)
            if not chunk:
                break
            observed += len(chunk)
            hasher.update(chunk)
    finally:
        body.close()

    if observed != byte_size or hasher.hexdigest() != digest:
        raise RuntimeError(f"immutable RunPod object failed byte verification: {key}")
    return True


def main() -> None:
    bootstrap.RunPodContentAddressedStore._verify_key = _verify_key
    bootstrap.main()


if __name__ == "__main__":
    main()
