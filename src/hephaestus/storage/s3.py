"""Optional S3-compatible immutable artifact adapter."""

from __future__ import annotations

import hashlib
import io
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable

from hephaestus.infrastructure.capabilities import OptionalCapabilityError
from hephaestus.infrastructure.observability import (
    EventSink,
    NullEventSink,
    StructuredEvent,
    emit_safely,
)

from .base import ArtifactRecord
from .filesystem import ArtifactIntegrityError, _normalized_expected_hash


class ArtifactUploadCancelled(RuntimeError):
    pass


def _not_found(exc: Exception) -> bool:
    if isinstance(exc, KeyError):
        return True
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error", {})
        if isinstance(error, dict) and str(error.get("Code")) in {"404", "NoSuchKey", "NotFound"}:
            return True
    return False


@dataclass(slots=True)
class S3ArtifactStore:
    """Content-addressed S3 adapter with invisible incomplete multipart uploads.

    S3 object creation is atomic at the final key. Multipart parts are not visible
    until ``complete_multipart_upload`` succeeds; failures are explicitly aborted.
    The injected client keeps cloud SDKs optional and makes conformance testing local.
    """

    client: Any
    bucket: str
    prefix: str = "hephaestus"
    multipart_threshold_bytes: int = 8 * 1024 * 1024
    multipart_part_bytes: int = 8 * 1024 * 1024
    server_side_encryption: str | None = "AES256"
    kms_key_id: str | None = None
    event_sink: EventSink = field(default_factory=NullEventSink)

    def __post_init__(self) -> None:
        if not self.bucket.strip():
            raise ValueError("S3 bucket must not be empty")
        if self.multipart_threshold_bytes <= 0 or self.multipart_part_bytes < 5 * 1024 * 1024:
            raise ValueError("multipart sizes must be positive and parts must be at least 5 MiB")
        self.prefix = self.prefix.strip("/")

    @classmethod
    def from_boto3(cls, bucket: str, **kwargs: object) -> "S3ArtifactStore":
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise OptionalCapabilityError(
                "S3 artifact support requires the 's3' optional dependencies"
            ) from exc
        client_kwargs = kwargs.pop("client_kwargs", {})
        if not isinstance(client_kwargs, dict):
            raise ValueError("client_kwargs must be a mapping")
        return cls(boto3.client("s3", **client_kwargs), bucket, **kwargs)

    def _key(self, digest: str) -> str:
        suffix = f"objects/sha256/{digest[:2]}/{digest}"
        return f"{self.prefix}/{suffix}" if self.prefix else suffix

    def _digest_from_ref(self, artifact_ref: str) -> str:
        prefix = "sha256:"
        digest = artifact_ref.removeprefix(prefix)
        if not artifact_ref.startswith(prefix) or len(digest) != 64:
            raise ValueError(f"invalid immutable artifact reference: {artifact_ref!r}")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError(f"invalid immutable artifact reference: {artifact_ref!r}") from exc
        return digest.lower()

    def _encryption(self) -> dict[str, object]:
        values: dict[str, object] = {}
        if self.server_side_encryption:
            values["ServerSideEncryption"] = self.server_side_encryption
        if self.kms_key_id:
            values["SSEKMSKeyId"] = self.kms_key_id
        return values

    def _metadata(self, digest: str, media_type: str | None) -> dict[str, object]:
        values: dict[str, object] = {
            "Metadata": {"sha256": digest},
            **self._encryption(),
        }
        if media_type:
            values["ContentType"] = media_type
        return values

    def _head(self, digest: str) -> dict[str, object] | None:
        try:
            return self.client.head_object(Bucket=self.bucket, Key=self._key(digest))
        except Exception as exc:
            if _not_found(exc):
                return None
            raise

    def _existing_record(
        self, digest: str, media_type: str | None = None
    ) -> ArtifactRecord | None:
        head = self._head(digest)
        if head is None:
            return None
        metadata = head.get("Metadata", {})
        if not isinstance(metadata, dict) or metadata.get("sha256") != digest:
            raise ArtifactIntegrityError(
                f"immutable S3 key exists without matching sha256 metadata: {self._key(digest)}"
            )
        return ArtifactRecord(
            artifact_ref=f"sha256:{digest}",
            content_hash=digest,
            hash_algorithm="sha256",
            byte_size=int(head.get("ContentLength", 0)),
            storage_path=f"s3://{self.bucket}/{self._key(digest)}",
            created_at=datetime.now(timezone.utc),
            media_type=media_type or head.get("ContentType"),
        )

    def _record(self, digest: str, byte_size: int, media_type: str | None) -> ArtifactRecord:
        return ArtifactRecord(
            artifact_ref=f"sha256:{digest}",
            content_hash=digest,
            hash_algorithm="sha256",
            byte_size=byte_size,
            storage_path=f"s3://{self.bucket}/{self._key(digest)}",
            created_at=datetime.now(timezone.utc),
            media_type=media_type,
        )

    def _emit(self, event_type: str, digest: str, **attributes: object) -> None:
        emit_safely(
            self.event_sink,
            StructuredEvent.create(
                event_type,
                "s3_artifact_store",
                entity_id=f"sha256:{digest}",
                severity="error" if event_type.endswith("failed") else "info",
                attributes=dict(attributes),
            ),
        )

    def put_bytes(
        self,
        data: bytes,
        *,
        expected_hash: str | None = None,
        media_type: str | None = None,
    ) -> ArtifactRecord:
        digest = hashlib.sha256(data).hexdigest()
        expected = _normalized_expected_hash(expected_hash)
        if expected is not None and expected != digest:
            raise ArtifactIntegrityError(f"expected sha256 {expected}, computed {digest}")
        existing = self._existing_record(digest, media_type)
        if existing is not None:
            return existing
        started = time.monotonic()
        arguments = {
            "Bucket": self.bucket,
            "Key": self._key(digest),
            "Body": data,
            "IfNoneMatch": "*",
            **self._metadata(digest, media_type),
        }
        try:
            self.client.put_object(**arguments)
        except Exception:
            # A concurrent identical writer may have won. Only accept it after
            # metadata validation; never overwrite an unexpected object.
            existing = self._existing_record(digest, media_type)
            if existing is None:
                raise
            return existing
        record = self._record(digest, len(data), media_type)
        self._emit(
            "storage.artifact_uploaded",
            digest,
            byte_size=len(data),
            duration_seconds=time.monotonic() - started,
        )
        return record

    @staticmethod
    def _hash_file(source: Path) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        return digest.hexdigest(), size

    def put_file(
        self,
        source: Path,
        *,
        expected_hash: str | None = None,
        media_type: str | None = None,
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> ArtifactRecord:
        digest, byte_size = self._hash_file(source)
        expected = _normalized_expected_hash(expected_hash)
        if expected is not None and expected != digest:
            raise ArtifactIntegrityError(f"expected sha256 {expected}, computed {digest}")
        existing = self._existing_record(digest, media_type)
        if existing is not None:
            return existing
        if byte_size < self.multipart_threshold_bytes:
            started = time.monotonic()
            try:
                with source.open("rb") as body:
                    self.client.put_object(
                        Bucket=self.bucket,
                        Key=self._key(digest),
                        Body=body,
                        IfNoneMatch="*",
                        **self._metadata(digest, media_type),
                    )
            except Exception:
                existing = self._existing_record(digest, media_type)
                if existing is None:
                    raise
                return existing
            record = self._record(digest, byte_size, media_type)
            self._emit(
                "storage.artifact_uploaded",
                digest,
                byte_size=byte_size,
                duration_seconds=time.monotonic() - started,
            )
            return record
        return self._multipart_upload(
            source,
            digest,
            byte_size,
            media_type,
            cancellation_requested or (lambda: False),
        )

    def _multipart_upload(
        self,
        source: Path,
        digest: str,
        byte_size: int,
        media_type: str | None,
        cancellation_requested: Callable[[], bool],
    ) -> ArtifactRecord:
        started = time.monotonic()
        created = self.client.create_multipart_upload(
            Bucket=self.bucket,
            Key=self._key(digest),
            **self._metadata(digest, media_type),
        )
        upload_id = str(created["UploadId"])
        parts: list[dict[str, object]] = []
        try:
            with source.open("rb") as handle:
                part_number = 1
                while True:
                    if cancellation_requested():
                        raise ArtifactUploadCancelled(f"upload cancelled: {source.name}")
                    chunk = handle.read(self.multipart_part_bytes)
                    if not chunk:
                        break
                    response = self.client.upload_part(
                        Bucket=self.bucket,
                        Key=self._key(digest),
                        UploadId=upload_id,
                        PartNumber=part_number,
                        Body=chunk,
                    )
                    parts.append({"PartNumber": part_number, "ETag": response["ETag"]})
                    part_number += 1
            self.client.complete_multipart_upload(
                Bucket=self.bucket,
                Key=self._key(digest),
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
                IfNoneMatch="*",
            )
        except BaseException as exc:
            self.client.abort_multipart_upload(
                Bucket=self.bucket,
                Key=self._key(digest),
                UploadId=upload_id,
            )
            self._emit("storage.artifact_upload_failed", digest, upload_id=upload_id)
            if not isinstance(exc, ArtifactUploadCancelled):
                existing = self._existing_record(digest, media_type)
                if existing is not None:
                    return existing
            raise
        record = self._record(digest, byte_size, media_type)
        self._emit(
            "storage.artifact_uploaded",
            digest,
            byte_size=byte_size,
            multipart_parts=len(parts),
            duration_seconds=time.monotonic() - started,
        )
        return record

    def exists(self, artifact_ref: str) -> bool:
        return self._head(self._digest_from_ref(artifact_ref)) is not None

    def open(self, artifact_ref: str) -> BinaryIO:
        digest = self._digest_from_ref(artifact_ref)
        started = time.monotonic()
        response = self.client.get_object(Bucket=self.bucket, Key=self._key(digest))
        body = response["Body"]
        data = body.read()
        if hasattr(body, "close"):
            body.close()
        self._emit(
            "storage.artifact_downloaded",
            digest,
            byte_size=len(data),
            duration_seconds=time.monotonic() - started,
        )
        return io.BytesIO(data)

    def get_bytes(self, artifact_ref: str) -> bytes:
        with self.open(artifact_ref) as handle:
            return handle.read()

    def verify(self, artifact_ref: str) -> bool:
        digest = self._digest_from_ref(artifact_ref)
        try:
            with self.open(artifact_ref) as handle:
                computed = hashlib.sha256(handle.read()).hexdigest()
        except Exception:
            self._emit("storage.artifact_verification_failed", digest, reason="read_failed")
            return False
        verified = computed == digest
        self._emit(
            "storage.artifact_verified" if verified else "storage.artifact_verification_failed",
            digest,
            verified=verified,
        )
        return verified

    def cleanup_incomplete_uploads(self) -> int:
        response = self.client.list_multipart_uploads(
            Bucket=self.bucket,
            Prefix=f"{self.prefix}/objects/" if self.prefix else "objects/",
        )
        uploads = response.get("Uploads", []) or []
        cleaned = 0
        for upload in uploads:
            self.client.abort_multipart_upload(
                Bucket=self.bucket,
                Key=upload["Key"],
                UploadId=upload["UploadId"],
            )
            cleaned += 1
        return cleaned

    def presign_get(self, artifact_ref: str, *, expires_seconds: int = 300) -> str:
        if expires_seconds <= 0:
            raise ValueError("expires_seconds must be positive")
        digest = self._digest_from_ref(artifact_ref)
        return str(
            self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": self._key(digest)},
                ExpiresIn=expires_seconds,
            )
        )
