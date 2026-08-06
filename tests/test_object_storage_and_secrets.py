from __future__ import annotations

import hashlib
import io
import json
import os

import pytest

from hephaestus.infrastructure.observability import StructuredEvent
from hephaestus.infrastructure.secrets import (
    FileMountedSecretsProvider,
    InjectedSecretsProvider,
    SecretReference,
    SecretResolutionError,
)
from hephaestus.storage import ArtifactUploadCancelled, S3ArtifactStore
from hephaestus.storage.filesystem import ArtifactIntegrityError


class FakeS3Client:
    def __init__(self) -> None:
        self.objects = {}
        self.uploads = {}
        self.aborted = []
        self._next_upload = 1

    def head_object(self, *, Bucket, Key):
        del Bucket
        if Key not in self.objects:
            raise KeyError(Key)
        item = self.objects[Key]
        return {
            "ContentLength": len(item["Body"]),
            "Metadata": dict(item["Metadata"]),
            "ContentType": item.get("ContentType"),
        }

    def put_object(self, *, Bucket, Key, Body, IfNoneMatch=None, **metadata):
        del Bucket
        if IfNoneMatch == "*" and Key in self.objects:
            raise RuntimeError("precondition failed")
        data = Body.read() if hasattr(Body, "read") else bytes(Body)
        self.objects[Key] = {"Body": data, **metadata}

    def get_object(self, *, Bucket, Key):
        del Bucket
        return {"Body": io.BytesIO(self.objects[Key]["Body"])}

    def create_multipart_upload(self, *, Bucket, Key, **metadata):
        del Bucket
        upload_id = f"upload-{self._next_upload}"
        self._next_upload += 1
        self.uploads[upload_id] = {"Key": Key, "Parts": {}, **metadata}
        return {"UploadId": upload_id}

    def upload_part(self, *, Bucket, Key, UploadId, PartNumber, Body):
        del Bucket, Key
        self.uploads[UploadId]["Parts"][PartNumber] = bytes(Body)
        return {"ETag": f"etag-{PartNumber}"}

    def complete_multipart_upload(
        self, *, Bucket, Key, UploadId, MultipartUpload, IfNoneMatch=None
    ):
        del Bucket, MultipartUpload
        if IfNoneMatch == "*" and Key in self.objects:
            raise RuntimeError("precondition failed")
        upload = self.uploads.pop(UploadId)
        body = b"".join(upload["Parts"][index] for index in sorted(upload["Parts"]))
        self.objects[Key] = {
            "Body": body,
            "Metadata": upload.get("Metadata", {}),
            "ContentType": upload.get("ContentType"),
        }

    def abort_multipart_upload(self, *, Bucket, Key, UploadId):
        del Bucket, Key
        self.uploads.pop(UploadId, None)
        self.aborted.append(UploadId)

    def list_multipart_uploads(self, *, Bucket, Prefix):
        del Bucket
        return {
            "Uploads": [
                {"Key": value["Key"], "UploadId": upload_id}
                for upload_id, value in self.uploads.items()
                if value["Key"].startswith(Prefix)
            ]
        }

    def generate_presigned_url(self, operation, *, Params, ExpiresIn):
        return f"fake://{operation}/{Params['Bucket']}/{Params['Key']}?expires={ExpiresIn}"


def test_s3_adapter_is_content_addressed_and_hash_verified(tmp_path) -> None:
    client = FakeS3Client()
    store = S3ArtifactStore(client, "bucket")
    data = b"immutable artifact"
    digest = hashlib.sha256(data).hexdigest()

    first = store.put_bytes(data, expected_hash=digest, media_type="text/plain")
    duplicate = store.put_bytes(data)

    assert first.artifact_ref == f"sha256:{digest}"
    assert duplicate.artifact_ref == first.artifact_ref
    assert store.exists(first.artifact_ref) is True
    assert store.get_bytes(first.artifact_ref) == data
    assert store.verify(first.artifact_ref) is True
    assert store.presign_get(first.artifact_ref).startswith("fake://get_object/")
    with pytest.raises(ArtifactIntegrityError):
        store.put_bytes(data, expected_hash="0" * 64)


def test_s3_multipart_completion_and_interruption_cleanup(tmp_path) -> None:
    client = FakeS3Client()
    store = S3ArtifactStore(
        client,
        "bucket",
        multipart_threshold_bytes=1,
        multipart_part_bytes=5 * 1024 * 1024,
    )
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"multipart payload")

    completed = store.put_file(source)
    assert store.get_bytes(completed.artifact_ref) == b"multipart payload"

    cancelled_source = tmp_path / "cancelled.bin"
    cancelled_source.write_bytes(b"cancelled payload")
    with pytest.raises(ArtifactUploadCancelled):
        store.put_file(cancelled_source, cancellation_requested=lambda: True)
    assert client.uploads == {}
    assert client.aborted

    dangling = client.create_multipart_upload(
        Bucket="bucket",
        Key="hephaestus/objects/sha256/aa/dangling",
        Metadata={"sha256": "dangling"},
    )
    assert dangling["UploadId"] in client.uploads
    assert store.cleanup_incomplete_uploads() == 1
    assert client.uploads == {}


def test_file_mounted_secret_requires_strict_regular_file(tmp_path) -> None:
    secret = tmp_path / "database_password"
    secret.write_text("sensitive-value\n", encoding="utf-8")
    secret.chmod(0o600)
    provider = FileMountedSecretsProvider(tmp_path)
    reference = SecretReference("file", "database_password")

    assert provider.resolve(reference) == "sensitive-value"
    assert "sensitive-value" not in json.dumps(reference.to_dict())
    secret.chmod(0o644)
    with pytest.raises(SecretResolutionError, match="permissions"):
        provider.resolve(reference)

    if hasattr(os, "symlink"):
        target = tmp_path / "target"
        target.write_text("value", encoding="utf-8")
        target.chmod(0o600)
        link = tmp_path / "link"
        link.symlink_to(target)
        with pytest.raises(SecretResolutionError, match="non-symlink"):
            provider.resolve(SecretReference("file", "link"))


def test_injected_secret_provider_and_events_never_expose_named_secret_values() -> None:
    provider = InjectedSecretsProvider("vault", lambda key: f"resolved-{key}")
    reference = SecretReference("vault", "training/key")
    assert provider.resolve(reference) == "resolved-training/key"

    event = StructuredEvent.create(
        "test.event",
        "test",
        attributes={
            "access_token": "must-not-appear",
            "payload": "x" * 1000,
            "safe": 1,
        },
    )
    assert event.attributes["access_token"] == "[redacted]"
    assert len(event.attributes["payload"]) == 512
    assert "must-not-appear" not in json.dumps(event.to_dict())
