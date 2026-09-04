"""Verified handoff from remote acquisition receipts into preprocessing inputs."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

from hephaestus.schemas.discovery_contract import DatasetCandidate, DatasetSelectionDecision
from hephaestus.utils.hashing import hash_file

from .acquisition import DatasetAcquisitionApproval, LocalAcquisition
from .acquisition_models import AcquisitionReceipt, AcquiredFileEvidence


_RECORD_FORMATS = {
    ".jsonl": "jsonl",
    ".json": "json",
    ".csv": "csv",
    ".parquet": "parquet",
}
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


def _normalized_hash(value: str | None) -> str | None:
    if value is None:
        return None
    digest = str(value).strip().removeprefix("sha256:").lower()
    if len(digest) != 64:
        raise ValueError("remote acquisition evidence contains an invalid sha256 value")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError("remote acquisition evidence contains an invalid sha256 value") from exc
    return digest


def _selected_file(
    receipt: AcquisitionReceipt,
    relative_path: str | None,
) -> AcquiredFileEvidence:
    files = tuple(receipt.acquired_files)
    if not files:
        raise ValueError("completed acquisition receipt contains no acquired files")
    if relative_path is None:
        if len(files) != 1:
            raise ValueError(
                "remote acquisition contains multiple files; relative_path must select one preprocessing input"
            )
        return files[0]
    requested = PurePosixPath(relative_path).as_posix()
    matches = [
        item
        for item in files
        if PurePosixPath(item.relative_path).as_posix() == requested
    ]
    if len(matches) != 1:
        raise ValueError(
            "requested preprocessing file is not present exactly once in the acquisition receipt"
        )
    return matches[0]


def validate_remote_acquisition_for_preprocessing(
    *,
    candidate: DatasetCandidate,
    selection: DatasetSelectionDecision,
    approval: DatasetAcquisitionApproval,
    receipt: AcquisitionReceipt,
    max_bytes: int,
    relative_path: str | None = None,
) -> LocalAcquisition:
    """Return a verified local view of one immutable remotely acquired file.

    The cache file is only a transport/materialization detail. The returned
    revision, hash, approval evidence, and caller-supplied candidate remain bound
    to the original provider identity; this function never rewrites a remote
    candidate into ``local_fixture`` provenance.
    """

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if (
        selection.status != "selected"
        or candidate.candidate_id not in selection.selected_candidate_ids
    ):
        raise PermissionError(
            "candidate is not selected by the supplied selection decision"
        )
    if approval.selection_decision_id != selection.decision_id:
        raise PermissionError(
            "approval does not reference the supplied selection decision"
        )
    approval_refs = tuple(
        sorted(
            {
                str(ref).strip()
                for ref in approval.approval_refs
                if str(ref).strip()
            }
        )
    )
    if candidate.candidate_id not in approval.approved_candidate_ids or not approval_refs:
        raise PermissionError("explicit dataset acquisition approval is missing")

    if receipt.completion_status != "completed":
        raise ValueError("remote acquisition receipt is not completed")
    if receipt.selection_decision_id != selection.decision_id:
        raise ValueError(
            "acquisition receipt selection decision does not match preprocessing input"
        )
    if receipt.candidate_id != candidate.candidate_id:
        raise ValueError(
            "acquisition receipt candidate does not match preprocessing input"
        )
    if receipt.provider_id != candidate.provider_id:
        raise ValueError(
            "acquisition receipt provider does not match candidate provenance"
        )
    if receipt.dataset_id != candidate.dataset_id:
        raise ValueError(
            "acquisition receipt dataset does not match candidate provenance"
        )
    receipt_approvals = tuple(
        sorted(
            {
                str(ref).strip()
                for ref in receipt.approval_refs
                if str(ref).strip()
            }
        )
    )
    if receipt_approvals != approval_refs:
        raise ValueError(
            "acquisition receipt approvals do not match preprocessing approval evidence"
        )

    candidate_revision = str(candidate.revision or "").strip().lower()
    requested_revision = str(receipt.requested_revision or "").strip().lower()
    resolved_revision = str(receipt.resolved_revision or "").strip().lower()
    if not resolved_revision or not _IMMUTABLE_REVISION.fullmatch(resolved_revision):
        raise ValueError(
            "acquisition receipt is missing a full immutable resolved revision"
        )
    if candidate_revision:
        if _IMMUTABLE_REVISION.fullmatch(candidate_revision):
            if candidate_revision != resolved_revision:
                raise ValueError(
                    "acquisition receipt revision does not match discovered candidate revision"
                )
        elif candidate_revision != requested_revision:
            raise ValueError(
                "acquisition receipt requested revision does not match discovered candidate revision"
            )
    provenance_revision = str(candidate.provenance.get("sha") or "").strip().lower()
    if provenance_revision and provenance_revision != resolved_revision:
        raise ValueError(
            "candidate provenance revision does not match acquisition receipt"
        )

    acquired = _selected_file(receipt, relative_path)
    source_path = Path(acquired.cache_ref).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"verified acquisition cache file does not exist: {source_path}"
        )
    observed_size = source_path.stat().st_size
    if observed_size != acquired.size_bytes:
        raise ValueError("acquired file byte size does not match receipt evidence")
    if observed_size > max_bytes:
        raise ValueError(
            f"dataset artifact exceeds preprocessing limit: {observed_size}>{max_bytes}"
        )

    expected_hash = _normalized_hash(acquired.local_content_hash)
    if expected_hash is None:
        raise ValueError(
            "acquisition receipt is missing the authoritative local sha256"
        )
    computed_hash = hash_file(source_path)
    if computed_hash != expected_hash:
        raise ValueError(
            "acquired file failed sha256 verification before preprocessing"
        )
    artifact_store_hash = _normalized_hash(acquired.artifact_store_content_hash)
    if artifact_store_hash is not None and artifact_store_hash != expected_hash:
        raise ValueError(
            "artifact-store hash disagrees with the verified acquisition sha256"
        )

    suffix = PurePosixPath(acquired.relative_path).suffix.casefold()
    record_format = _RECORD_FORMATS.get(suffix)
    if record_format is None:
        raise ValueError(
            f"unsupported remotely acquired preprocessing format: {suffix or '<none>'}"
        )

    return LocalAcquisition(
        candidate_id=candidate.candidate_id,
        source_path=source_path,
        source_content_hash=f"sha256:{computed_hash}",
        source_bytes=observed_size,
        record_format=record_format,
        revision=resolved_revision,
        approval_refs=approval_refs,
    )


__all__ = ["validate_remote_acquisition_for_preprocessing"]
