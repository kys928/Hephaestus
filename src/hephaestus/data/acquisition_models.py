"""JSON-safe records for planned and completed remote dataset acquisition."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from hephaestus.providers.datasets.acquisition import (
    ProviderDatasetFile,
    ProviderDatasetSnapshot,
)
from hephaestus.schemas.contract_common import ContractIssue
from hephaestus.utils.hashing import hash_json


@dataclass(frozen=True, slots=True)
class RemoteAcquisitionLimits:
    max_bytes: int = 4 * 1024 * 1024 * 1024
    max_files: int = 1_000
    chunk_size: int = 1024 * 1024
    timeout_seconds: float = 30.0
    disk_reserve_bytes: int = 64 * 1024 * 1024
    allowed_suffixes: tuple[str, ...] = (
        ".jsonl",
        ".json",
        ".csv",
        ".parquet",
        ".arrow",
        ".txt",
        ".gz",
        ".zst",
        ".zip",
    )

    def __post_init__(self) -> None:
        if self.max_bytes <= 0 or self.max_files <= 0 or self.chunk_size <= 0:
            raise ValueError(
                "acquisition byte, file, and chunk limits must be positive"
            )
        if self.timeout_seconds <= 0 or self.disk_reserve_bytes < 0:
            raise ValueError("timeout must be positive and disk reserve non-negative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RemoteAcquisitionPlan:
    plan_id: str
    selection_decision_id: str
    candidate_id: str
    snapshot: ProviderDatasetSnapshot
    files: tuple[ProviderDatasetFile, ...]
    limits: RemoteAcquisitionLimits
    required_approvals: tuple[str, ...] = ()
    authentication_reference: dict[str, str] | None = None
    estimated_total_bytes: int | None = None
    warnings: tuple[str, ...] = ()
    hash_algorithm: str = "sha256"
    plan_version: str = "dataset-acquisition-plan.v1"

    @classmethod
    def create(
        cls,
        *,
        selection_decision_id: str,
        candidate_id: str,
        snapshot: ProviderDatasetSnapshot,
        files: tuple[ProviderDatasetFile, ...],
        limits: RemoteAcquisitionLimits,
        required_approvals: tuple[str, ...] = (),
        authentication_reference: dict[str, str] | None = None,
        warnings: tuple[str, ...] = (),
    ) -> RemoteAcquisitionPlan:
        estimated = (
            sum(file.size_bytes or 0 for file in files)
            if all(file.size_bytes is not None for file in files)
            else None
        )
        seed = {
            "selection_decision_id": selection_decision_id,
            "candidate_id": candidate_id,
            "snapshot": snapshot.to_dict(),
            "files": [file.to_dict() for file in files],
            "limits": limits.to_dict(),
            "required_approvals": list(required_approvals),
            "authentication_reference": authentication_reference,
            "estimated_total_bytes": estimated,
            "warnings": list(warnings),
            "plan_version": "dataset-acquisition-plan.v1",
        }
        return cls(
            plan_id=f"dataset-acquisition-plan-{hash_json(seed)[:24]}",
            selection_decision_id=selection_decision_id,
            candidate_id=candidate_id,
            snapshot=snapshot,
            files=files,
            limits=limits,
            required_approvals=required_approvals,
            authentication_reference=authentication_reference,
            estimated_total_bytes=estimated,
            warnings=warnings,
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        return payload


@dataclass(frozen=True, slots=True)
class AcquisitionPlanningResult:
    status: str
    plan: RemoteAcquisitionPlan | None = None
    issues: tuple[ContractIssue, ...] = ()


@dataclass(frozen=True, slots=True)
class AcquiredFileEvidence:
    relative_path: str
    source_url: str
    provider_object_id: str | None
    size_bytes: int
    provider_declared_hash: str | None
    provider_hash_algorithm: str | None
    provider_hash_status: str
    transport_checksum: str | None
    transport_checksum_status: str
    local_content_hash: str
    cache_key: str
    cache_status: str
    cache_ref: str
    artifact_ref: str | None = None
    artifact_store_content_hash: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TransferAttempt:
    relative_path: str
    attempt: int
    action: str
    status: str
    requested_offset: int = 0
    transferred_bytes: int = 0
    response_status: int | None = None
    range_supported: bool | None = None
    partial_preserved: bool = False
    issue_code: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AcquisitionReceipt:
    receipt_id: str
    plan_id: str
    selection_decision_id: str
    approval_refs: tuple[str, ...]
    candidate_id: str
    provider_id: str
    dataset_id: str
    requested_revision: str
    resolved_revision: str
    acquired_files: tuple[AcquiredFileEvidence, ...]
    byte_totals: dict[str, int]
    cache_status: str
    dataset_card_ref: str | None
    dataset_card_revision: str | None
    license: str | None
    license_source: str | None
    transfer_attempts: tuple[TransferAttempt, ...]
    partial_recovery_evidence: tuple[dict[str, object], ...]
    artifact_refs: tuple[str, ...]
    warnings: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    cleanup: tuple[dict[str, object], ...]
    completion_status: str
    issues: tuple[ContractIssue, ...] = ()
    observations: dict[str, str] = field(default_factory=dict)
    receipt_version: str = "dataset-acquisition-receipt.v1"

    @classmethod
    def create(cls, **values: Any) -> AcquisitionReceipt:
        observations = dict(values.pop("observations", {}))
        seed = {
            key: _json_value(value)
            for key, value in values.items()
            if key not in {"receipt_id", "observations"}
        }
        seed["receipt_version"] = "dataset-acquisition-receipt.v1"
        return cls(
            receipt_id=f"dataset-acquisition-receipt-{hash_json(seed)[:24]}",
            observations=observations,
            **values,
        )

    def deterministic_dict(self) -> dict[str, object]:
        payload = self.to_dict()
        payload.pop("observations", None)
        payload.pop("receipt_id", None)
        return payload

    def to_dict(self) -> dict[str, object]:
        return {
            "receipt_id": self.receipt_id,
            "plan_id": self.plan_id,
            "selection_decision_id": self.selection_decision_id,
            "approval_refs": list(self.approval_refs),
            "candidate_id": self.candidate_id,
            "provider_id": self.provider_id,
            "dataset_id": self.dataset_id,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "acquired_files": [item.to_dict() for item in self.acquired_files],
            "byte_totals": dict(self.byte_totals),
            "cache_status": self.cache_status,
            "dataset_card_ref": self.dataset_card_ref,
            "dataset_card_revision": self.dataset_card_revision,
            "license": self.license,
            "license_source": self.license_source,
            "transfer_attempts": [item.to_dict() for item in self.transfer_attempts],
            "partial_recovery_evidence": [
                dict(item) for item in self.partial_recovery_evidence
            ],
            "artifact_refs": list(self.artifact_refs),
            "warnings": list(self.warnings),
            "missing_evidence": list(self.missing_evidence),
            "cleanup": [dict(item) for item in self.cleanup],
            "completion_status": self.completion_status,
            "issues": [issue.to_dict() for issue in self.issues],
            "observations": dict(self.observations),
            "receipt_version": self.receipt_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AcquisitionReceipt:
        return cls(
            receipt_id=str(payload["receipt_id"]),
            plan_id=str(payload["plan_id"]),
            selection_decision_id=str(payload["selection_decision_id"]),
            approval_refs=tuple(map(str, payload.get("approval_refs", []))),
            candidate_id=str(payload["candidate_id"]),
            provider_id=str(payload["provider_id"]),
            dataset_id=str(payload["dataset_id"]),
            requested_revision=str(payload["requested_revision"]),
            resolved_revision=str(payload["resolved_revision"]),
            acquired_files=tuple(
                AcquiredFileEvidence(**dict(item))
                for item in payload.get("acquired_files", [])
            ),
            byte_totals={
                str(key): int(value)
                for key, value in dict(payload.get("byte_totals", {})).items()
            },
            cache_status=str(payload.get("cache_status", "miss")),
            dataset_card_ref=payload.get("dataset_card_ref"),
            dataset_card_revision=payload.get("dataset_card_revision"),
            license=payload.get("license"),
            license_source=payload.get("license_source"),
            transfer_attempts=tuple(
                TransferAttempt(**dict(item))
                for item in payload.get("transfer_attempts", [])
            ),
            partial_recovery_evidence=tuple(
                dict(item) for item in payload.get("partial_recovery_evidence", [])
            ),
            artifact_refs=tuple(map(str, payload.get("artifact_refs", []))),
            warnings=tuple(map(str, payload.get("warnings", []))),
            missing_evidence=tuple(map(str, payload.get("missing_evidence", []))),
            cleanup=tuple(dict(item) for item in payload.get("cleanup", [])),
            completion_status=str(payload.get("completion_status", "failed")),
            issues=tuple(
                ContractIssue.from_dict(dict(item))
                for item in payload.get("issues", [])
            ),
            observations={
                str(key): str(value)
                for key, value in dict(payload.get("observations", {})).items()
            },
            receipt_version=str(
                payload.get("receipt_version", "dataset-acquisition-receipt.v1")
            ),
        )


@dataclass(frozen=True, slots=True)
class RemoteAcquisitionResult:
    receipt: AcquisitionReceipt

    @property
    def completed(self) -> bool:
        return self.receipt.completion_status == "completed"


def _json_value(value: object) -> object:
    if hasattr(value, "to_dict"):
        return value.to_dict()  # type: ignore[no-any-return, union-attr]
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


__all__ = [
    "AcquiredFileEvidence",
    "AcquisitionPlanningResult",
    "AcquisitionReceipt",
    "RemoteAcquisitionLimits",
    "RemoteAcquisitionPlan",
    "RemoteAcquisitionResult",
    "TransferAttempt",
]
