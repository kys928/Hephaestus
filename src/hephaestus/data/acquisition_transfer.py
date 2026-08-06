"""Streaming, checksum, cache-finalization, and artifact-store transfer engine."""

from __future__ import annotations

import base64
import hashlib
import re
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from hephaestus.providers.datasets.acquisition import ProviderDatasetFile
from hephaestus.schemas.contract_common import ContractIssue
from hephaestus.storage.base import ArtifactStore
from hephaestus.utils.hashing import hash_file

from .acquisition_cache import DatasetAcquisitionCache
from .acquisition_models import (
    AcquiredFileEvidence,
    RemoteAcquisitionPlan,
    TransferAttempt,
)
from .acquisition_transport import AcquisitionTransportError, DownloadTransport


@dataclass(frozen=True, slots=True)
class TransferOutcome:
    evidence: AcquiredFileEvidence | None
    attempt: TransferAttempt
    recovery: tuple[dict[str, object], ...] = ()
    cleanup: tuple[dict[str, object], ...] = ()
    issue: ContractIssue | None = None
    transferred_bytes: int = 0


@dataclass(slots=True)
class AcquisitionFileTransfer:
    cache: DatasetAcquisitionCache
    transport: DownloadTransport
    artifact_store: ArtifactStore | None = None

    def transfer_file(
        self,
        plan: RemoteAcquisitionPlan,
        file: ProviderDatasetFile,
        *,
        attempt_number: int,
        token: str | None,
        acquired_bytes: int,
        cancellation_requested: Callable[[], bool] | None,
    ) -> TransferOutcome:
        partial = self.cache.load_partial(
            provider_id=plan.snapshot.provider_id,
            dataset_id=plan.snapshot.dataset_id,
            resolved_revision=plan.snapshot.resolved_revision,
            file=file,
        )
        recovery: list[dict[str, object]] = []
        cleanup: list[dict[str, object]] = []
        offset = partial.byte_count if partial else 0
        headers: dict[str, str] = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if offset:
            headers["Range"] = f"bytes={offset}-"
            if partial and partial.etag:
                headers["If-Range"] = partial.etag
        try:
            response = self.transport.open(
                file.source_url,
                headers=headers,
                timeout_seconds=plan.limits.timeout_seconds,
            )
        except AcquisitionTransportError as exc:
            return TransferOutcome(
                None,
                TransferAttempt(
                    file.relative_path,
                    attempt_number,
                    "resume" if offset else "download",
                    "failed",
                    requested_offset=offset,
                    partial_preserved=bool(partial),
                    issue_code=exc.code,
                ),
                issue=ContractIssue(
                    code=exc.code,
                    category="provider_unavailable"
                    if exc.retryable
                    else "runtime_failure",
                    message=str(exc),
                    retryable=exc.retryable,
                    blocking=True,
                ),
            )

        response_headers = {
            str(key).casefold(): str(value) for key, value in response.headers.items()
        }
        response_etag = response_headers.get("etag") or response_headers.get(
            "x-linked-etag"
        )
        mode = "ab" if offset else "wb"
        range_supported: bool | None = None
        if offset:
            content_range = response_headers.get("content-range", "")
            if response.status == 206 and content_range.startswith(f"bytes {offset}-"):
                range_supported = True
                if (
                    partial
                    and partial.etag
                    and response_etag
                    and partial.etag != response_etag
                ):
                    response.close()
                    cleanup.append(self.cache.clear_partial_by_key(partial.cache_key))
                    issue = ContractIssue(
                        code="revision_changed",
                        category="artifact_integrity",
                        message="remote identity changed while resuming a partial transfer",
                        blocking=True,
                    )
                    return TransferOutcome(
                        None,
                        TransferAttempt(
                            file.relative_path,
                            attempt_number,
                            "resume",
                            "failed",
                            offset,
                            response_status=206,
                            range_supported=True,
                            issue_code=issue.code,
                        ),
                        cleanup=tuple(cleanup),
                        issue=issue,
                    )
                recovery.append(
                    {
                        "relative_path": file.relative_path,
                        "action": "range_resume",
                        "offset": offset,
                    }
                )
            elif response.status == 200:
                range_supported = False
                mode = "wb"
                recovery.append(
                    {
                        "relative_path": file.relative_path,
                        "action": "restart_from_zero",
                        "prior_offset": offset,
                    }
                )
                offset = 0
            else:
                response.close()
                if partial:
                    cleanup.append(self.cache.clear_partial_by_key(partial.cache_key))
                issue = ContractIssue(
                    code="invalid_resume_response",
                    category="artifact_integrity",
                    message="provider resume response did not match the requested byte offset",
                    blocking=True,
                )
                return TransferOutcome(
                    None,
                    TransferAttempt(
                        file.relative_path,
                        attempt_number,
                        "resume",
                        "failed",
                        offset,
                        response_status=response.status,
                        range_supported=False,
                        issue_code=issue.code,
                    ),
                    cleanup=tuple(cleanup),
                    issue=issue,
                )
        elif response.status not in {200, 206}:
            response.close()
            issue = ContractIssue(
                code="provider_transfer_error",
                category="provider_unavailable",
                message=f"provider returned unexpected transfer status {response.status}",
                blocking=True,
            )
            return TransferOutcome(
                None,
                TransferAttempt(
                    file.relative_path,
                    attempt_number,
                    "download",
                    "failed",
                    response_status=response.status,
                    issue_code=issue.code,
                ),
                issue=issue,
            )

        key = self.cache.cache_key(
            provider_id=plan.snapshot.provider_id,
            dataset_id=plan.snapshot.dataset_id,
            resolved_revision=plan.snapshot.resolved_revision,
            file=file,
        )
        state = self.cache.prepare_partial(
            provider_id=plan.snapshot.provider_id,
            dataset_id=plan.snapshot.dataset_id,
            resolved_revision=plan.snapshot.resolved_revision,
            file=file,
            byte_count=offset,
            etag=response_etag or (partial.etag if partial else file.etag),
        )
        expected_remaining = max(0, (file.size_bytes or 0) - offset)
        free_bytes = shutil.disk_usage(state.path.parent).free
        if (
            file.size_bytes is not None
            and free_bytes < expected_remaining + plan.limits.disk_reserve_bytes
        ):
            response.close()
            cleanup.append(self.cache.clear_partial_by_key(key))
            issue = ContractIssue(
                code="disk_space_failure",
                category="budget_exceeded",
                message="insufficient free disk space for bounded dataset transfer",
                blocking=True,
            )
            return TransferOutcome(
                None,
                TransferAttempt(
                    file.relative_path,
                    attempt_number,
                    "download",
                    "failed",
                    offset,
                    response_status=response.status,
                    issue_code=issue.code,
                ),
                cleanup=tuple(cleanup),
                issue=issue,
            )

        network_bytes = 0
        try:
            with state.path.open(mode) as handle:
                while True:
                    if cancellation_requested is not None and cancellation_requested():
                        raise _TransferStop(
                            "acquisition_cancelled",
                            "dataset acquisition was cancelled",
                            True,
                        )
                    chunk = response.read(plan.limits.chunk_size)
                    if not chunk:
                        break
                    handle.write(chunk)
                    network_bytes += len(chunk)
                    current_size = offset + network_bytes
                    if acquired_bytes + current_size > plan.limits.max_bytes:
                        raise _TransferStop(
                            "maximum_byte_budget_exceeded",
                            "dataset transfer exceeded the configured byte budget",
                            False,
                        )
        except _TransferStop as exc:
            current_size = state.path.stat().st_size if state.path.exists() else 0
            if exc.preserve_partial:
                self.cache.prepare_partial(
                    provider_id=plan.snapshot.provider_id,
                    dataset_id=plan.snapshot.dataset_id,
                    resolved_revision=plan.snapshot.resolved_revision,
                    file=file,
                    byte_count=current_size,
                    etag=response_etag or state.etag,
                )
            else:
                cleanup.append(self.cache.clear_partial_by_key(key))
            issue = ContractIssue(
                code=exc.code,
                category="runtime_failure"
                if exc.code == "acquisition_cancelled"
                else "budget_exceeded",
                message=str(exc),
                retryable=exc.preserve_partial,
                blocking=True,
            )
            return TransferOutcome(
                None,
                TransferAttempt(
                    file.relative_path,
                    attempt_number,
                    "resume" if offset else "download",
                    "partial",
                    offset,
                    network_bytes,
                    response.status,
                    range_supported,
                    exc.preserve_partial,
                    exc.code,
                ),
                cleanup=tuple(cleanup),
                issue=issue,
                transferred_bytes=network_bytes,
            )
        except (OSError, AcquisitionTransportError) as exc:
            current_size = state.path.stat().st_size if state.path.exists() else 0
            self.cache.prepare_partial(
                provider_id=plan.snapshot.provider_id,
                dataset_id=plan.snapshot.dataset_id,
                resolved_revision=plan.snapshot.resolved_revision,
                file=file,
                byte_count=current_size,
                etag=response_etag or state.etag,
            )
            code = (
                exc.code
                if isinstance(exc, AcquisitionTransportError)
                else "connection_interrupted"
            )
            issue = ContractIssue(
                code=code,
                category="provider_unavailable",
                message="remote dataset transfer was interrupted",
                retryable=True,
                blocking=True,
            )
            return TransferOutcome(
                None,
                TransferAttempt(
                    file.relative_path,
                    attempt_number,
                    "resume" if offset else "download",
                    "partial",
                    offset,
                    network_bytes,
                    response.status,
                    range_supported,
                    True,
                    code,
                ),
                issue=issue,
                transferred_bytes=network_bytes,
            )
        finally:
            response.close()

        byte_size = state.path.stat().st_size
        if file.size_bytes is not None and byte_size != file.size_bytes:
            self.cache.prepare_partial(
                provider_id=plan.snapshot.provider_id,
                dataset_id=plan.snapshot.dataset_id,
                resolved_revision=plan.snapshot.resolved_revision,
                file=file,
                byte_count=byte_size,
                etag=response_etag or state.etag,
            )
            issue = ContractIssue(
                code="partial_transfer",
                category="artifact_integrity",
                message=f"transfer ended at {byte_size} bytes; expected {file.size_bytes}",
                retryable=True,
                blocking=True,
            )
            return TransferOutcome(
                None,
                TransferAttempt(
                    file.relative_path,
                    attempt_number,
                    "resume" if offset else "download",
                    "partial",
                    offset,
                    network_bytes,
                    response.status,
                    range_supported,
                    True,
                    issue.code,
                ),
                issue=issue,
                transferred_bytes=network_bytes,
            )

        local_digest = hash_file(state.path)
        provider_hash_status = provider_hash_status_for(state.path, byte_size, file)
        if provider_hash_status == "mismatch":
            cleanup.append(self.cache.clear_partial_by_key(key))
            issue = ContractIssue(
                code="checksum_mismatch",
                category="artifact_integrity",
                message="provider-declared file hash did not match acquired content",
                blocking=True,
            )
            return TransferOutcome(
                None,
                TransferAttempt(
                    file.relative_path,
                    attempt_number,
                    "resume" if offset else "download",
                    "failed",
                    offset,
                    network_bytes,
                    response.status,
                    range_supported,
                    False,
                    issue.code,
                ),
                cleanup=tuple(cleanup),
                issue=issue,
                transferred_bytes=network_bytes,
            )
        transport_checksum, transport_status = transport_checksum_status(
            response_headers, local_digest
        )
        if transport_status == "mismatch":
            cleanup.append(self.cache.clear_partial_by_key(key))
            issue = ContractIssue(
                code="transport_checksum_mismatch",
                category="artifact_integrity",
                message="transport checksum did not match acquired content",
                blocking=True,
            )
            return TransferOutcome(
                None,
                TransferAttempt(
                    file.relative_path,
                    attempt_number,
                    "download",
                    "failed",
                    offset,
                    network_bytes,
                    response.status,
                    range_supported,
                    False,
                    issue.code,
                ),
                cleanup=tuple(cleanup),
                issue=issue,
                transferred_bytes=network_bytes,
            )
        try:
            cached = self.cache.store_completed(
                provider_id=plan.snapshot.provider_id,
                dataset_id=plan.snapshot.dataset_id,
                resolved_revision=plan.snapshot.resolved_revision,
                file=file,
                source=state.path,
                local_content_hash=f"sha256:{local_digest}",
                byte_size=byte_size,
            )
        except (OSError, ValueError) as exc:
            issue = ContractIssue(
                code="cache_finalization_failure",
                category="artifact_integrity",
                message=f"cache finalization failed with {type(exc).__name__}",
                blocking=True,
                metadata={"exception_type": type(exc).__name__},
            )
            return TransferOutcome(
                None,
                TransferAttempt(
                    file.relative_path,
                    attempt_number,
                    "cache_finalize",
                    "failed",
                    offset,
                    network_bytes,
                    response.status,
                    range_supported,
                    False,
                    issue.code,
                ),
                issue=issue,
                transferred_bytes=network_bytes,
            )
        assert cached.path is not None and cached.content_hash is not None
        artifact_ref, artifact_hash, artifact_issue = self.store_artifact(
            cached.path, cached.content_hash
        )
        if artifact_issue is not None:
            return TransferOutcome(
                None,
                TransferAttempt(
                    file.relative_path,
                    attempt_number,
                    "artifact_store",
                    "failed",
                    offset,
                    network_bytes,
                    response.status,
                    range_supported,
                    False,
                    artifact_issue.code,
                ),
                issue=artifact_issue,
                transferred_bytes=network_bytes,
            )
        evidence = AcquiredFileEvidence(
            relative_path=file.relative_path,
            source_url=file.source_url,
            provider_object_id=file.object_id,
            size_bytes=byte_size,
            provider_declared_hash=file.provider_hash,
            provider_hash_algorithm=file.provider_hash_algorithm,
            provider_hash_status=provider_hash_status,
            transport_checksum=transport_checksum,
            transport_checksum_status=transport_status,
            local_content_hash=cached.content_hash,
            cache_key=cached.cache_key,
            cache_status="miss_stored",
            cache_ref=str(cached.path),
            artifact_ref=artifact_ref,
            artifact_store_content_hash=artifact_hash,
        )
        return TransferOutcome(
            evidence,
            TransferAttempt(
                file.relative_path,
                attempt_number,
                "resume" if offset else "download",
                "completed",
                offset,
                network_bytes,
                response.status,
                range_supported,
            ),
            recovery=tuple(recovery),
            cleanup=tuple(cleanup),
            transferred_bytes=network_bytes,
        )

    def store_artifact(
        self, path: Path, local_content_hash: str
    ) -> tuple[str | None, str | None, ContractIssue | None]:
        if self.artifact_store is None:
            return None, None, None
        try:
            record = self.artifact_store.put_file(
                path, expected_hash=local_content_hash
            )
            if not self.artifact_store.verify(record.artifact_ref):
                raise ValueError("artifact store verification returned false")
            return (
                record.artifact_ref,
                f"{record.hash_algorithm}:{record.content_hash}",
                None,
            )
        except Exception as exc:  # noqa: BLE001 - injected artifact-store failure boundary
            return (
                None,
                None,
                ContractIssue(
                    code="artifact_store_failure",
                    category="artifact_integrity",
                    message=f"artifact store integration failed with {type(exc).__name__}",
                    blocking=True,
                    metadata={"exception_type": type(exc).__name__},
                ),
            )


class _TransferStop(RuntimeError):
    def __init__(self, code: str, message: str, preserve_partial: bool) -> None:
        super().__init__(message)
        self.code = code
        self.preserve_partial = preserve_partial


def provider_hash_status_for(
    path: Path, byte_size: int, file: ProviderDatasetFile
) -> str:
    if not file.provider_hash or not file.provider_hash_algorithm:
        return "not_provided"
    expected = file.provider_hash.removeprefix(
        f"{file.provider_hash_algorithm}:"
    ).lower()
    if file.provider_hash_algorithm == "sha256":
        computed = hash_file(path)
    elif file.provider_hash_algorithm == "git-blob-sha1":
        digest = hashlib.sha1()
        digest.update(f"blob {byte_size}\0".encode("ascii"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        computed = digest.hexdigest()
    else:
        return "provided_algorithm_not_supported"
    return "verified" if computed == expected else "mismatch"


def transport_checksum_status(
    headers: Mapping[str, str], local_sha256: str
) -> tuple[str | None, str]:
    hex_value = headers.get("x-checksum-sha256")
    if hex_value and re.fullmatch(r"[0-9a-fA-F]{64}", hex_value.strip()):
        normalized = hex_value.strip().lower()
        return (
            f"sha256:{normalized}",
            "verified" if normalized == local_sha256 else "mismatch",
        )
    digest = headers.get("digest", "")
    for part in digest.split(","):
        name, separator, value = part.strip().partition("=")
        if separator and name.casefold() in {"sha-256", "sha256"}:
            try:
                normalized = base64.b64decode(value.strip()).hex()
            except (ValueError, TypeError):
                return digest, "malformed"
            return (
                f"sha256:{normalized}",
                "verified" if normalized == local_sha256 else "mismatch",
            )
    return None, "not_provided"


__all__ = [
    "AcquisitionFileTransfer",
    "TransferOutcome",
    "provider_hash_status_for",
    "transport_checksum_status",
]
