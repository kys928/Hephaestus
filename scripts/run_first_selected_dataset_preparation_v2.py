#!/usr/bin/env python3
"""Run the selected-dataset preparation with RunPod single-PUT file staging.

The first live attempt proved immutable Hugging Face transfer of the selected
300 MB shard but exposed an integration-only incompatibility between boto3's
multipart ``upload_file`` helper and the RunPod S3 endpoint. The domain
acquisition service is unchanged. This wrapper replaces only the integration
artifact-store file writers with bounded streaming PutObject calls.
"""
from __future__ import annotations

from pathlib import Path

import run_first_selected_dataset_preparation as base
from hephaestus.storage.base import ArtifactRecord


def _put_file_single_request(
    self: base.RunPodS3ArtifactStore,
    source: Path,
    *,
    expected_hash: str | None = None,
    media_type: str | None = None,
) -> ArtifactRecord:
    source = Path(source)
    digest = base.hash_file(source)
    expected = str(expected_hash or "").removeprefix("sha256:")
    if expected and expected != digest:
        raise ValueError("expected content hash does not match file")
    size = source.stat().st_size
    key = self._key(digest)
    if self._head(key) is None:
        arguments: dict[str, object] = {
            "Bucket": self.bucket,
            "Key": key,
            "ContentLength": size,
            "Metadata": {"sha256": digest},
        }
        if media_type:
            arguments["ContentType"] = media_type
        with source.open("rb") as handle:
            arguments["Body"] = handle
            self.client.put_object(**arguments)
    if not self._verify_key(key, digest, size):
        raise RuntimeError(f"RunPod single-PUT file verification failed: {key}")
    return ArtifactRecord(
        artifact_ref=f"sha256:{digest}",
        content_hash=digest,
        hash_algorithm="sha256",
        byte_size=size,
        storage_path=f"s3://{self.bucket}/{key}",
        created_at=base.datetime.now(base.timezone.utc),
        media_type=media_type,
    )


def _stage_named_file_single_request(
    self: base.RunPodS3ArtifactStore,
    *,
    key: str,
    source: Path,
    media_type: str | None = None,
) -> dict[str, object]:
    source = Path(source)
    digest = base.hash_file(source)
    size = source.stat().st_size
    if self._head(key) is None:
        arguments: dict[str, object] = {
            "Bucket": self.bucket,
            "Key": key,
            "ContentLength": size,
            "Metadata": {"sha256": digest},
        }
        if media_type:
            arguments["ContentType"] = media_type
        with source.open("rb") as handle:
            arguments["Body"] = handle
            self.client.put_object(**arguments)
    if not self._verify_key(key, digest, size):
        raise RuntimeError(f"named RunPod single-PUT verification failed: {key}")
    return {
        "storage_path": f"s3://{self.bucket}/{key}",
        "sha256": f"sha256:{digest}",
        "byte_size": size,
        "key": key,
    }


base.RunPodS3ArtifactStore.put_file = _put_file_single_request  # type: ignore[method-assign]
base.RunPodS3ArtifactStore.stage_named_file = _stage_named_file_single_request  # type: ignore[method-assign]


if __name__ == "__main__":
    base.main()
