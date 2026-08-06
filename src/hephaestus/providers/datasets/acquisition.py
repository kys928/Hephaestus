"""Data-owned provider contracts for immutable remote dataset acquisition."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Protocol, runtime_checkable

from hephaestus.schemas.contract_common import ContractIssue


@dataclass(frozen=True, slots=True)
class ProviderDatasetFile:
    relative_path: str
    source_url: str
    size_bytes: int | None = None
    provider_hash: str | None = None
    provider_hash_algorithm: str | None = None
    etag: str | None = None
    object_id: str | None = None
    media_type: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProviderDatasetSnapshot:
    provider_id: str
    dataset_id: str
    requested_revision: str
    resolved_revision: str
    dataset_card_ref: str | None = None
    dataset_card_revision: str | None = None
    license: str | None = None
    license_source: str | None = None
    terms: tuple[str, ...] = ()
    citation: str | None = None
    authors: tuple[str, ...] = ()
    gated: bool = False
    private: bool = False
    remote_code_required: bool = False
    provenance_confidence: str = "provider_metadata"
    missing_metadata: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DatasetProviderAcquisitionError(RuntimeError):
    """Normalized provider failure that never includes secret values."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        category: str = "provider_unavailable",
        retryable: bool = False,
        blocking: bool = True,
        metadata: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.category = category
        self.retryable = retryable
        self.blocking = blocking
        self.metadata = dict(metadata or {})

    def to_issue(self, *, evidence_refs: Sequence[str] = ()) -> ContractIssue:
        return ContractIssue(
            code=self.code,
            category=self.category,
            message=str(self),
            retryable=self.retryable,
            blocking=self.blocking,
            evidence_refs=list(evidence_refs),
            metadata=self.metadata,
        )


@runtime_checkable
class RemoteDatasetAcquisitionProvider(Protocol):
    provider_id: str

    def resolve_revision(
        self,
        dataset_id: str,
        requested_revision: str,
        *,
        token: str | None = None,
    ) -> ProviderDatasetSnapshot: ...

    def enumerate_files(
        self,
        snapshot: ProviderDatasetSnapshot,
        *,
        token: str | None = None,
    ) -> Sequence[ProviderDatasetFile]: ...

    def revision_is_current(
        self,
        snapshot: ProviderDatasetSnapshot,
        *,
        token: str | None = None,
    ) -> bool: ...


__all__ = [
    "DatasetProviderAcquisitionError",
    "ProviderDatasetFile",
    "ProviderDatasetSnapshot",
    "RemoteDatasetAcquisitionProvider",
]
