from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hephaestus.schemas.discovery_contract import DatasetCandidate, DatasetSelectionDecision
from hephaestus.utils.hashing import hash_file
from hephaestus.utils.hashing import hash_json


@dataclass(frozen=True, slots=True)
class AcquiredDataset:
    dataset_id: str
    source_identity: str
    license: str
    quality_score: float
    total_examples: int | None = None
    risks: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_backend_payload(self) -> dict[str, object]:
        payload = {
            "dataset_id": self.dataset_id,
            "source_identity": self.source_identity,
            "license": self.license,
            "quality_score": self.quality_score,
            "total_examples": self.total_examples,
            "risks": list(self.risks),
            **self.metadata,
        }
        payload.setdefault("content_hash", hash_json(payload))
        payload.setdefault("hash_type", "sha256")
        return payload


def normalize_acquired_dataset(payload: dict[str, Any]) -> dict[str, object]:
    dataset_id = str(payload.get("dataset_id") or "dataset-unknown")
    source = str(payload.get("source_identity") or payload.get("source") or "unknown")
    risks = [str(risk) for risk in payload.get("risks", [])] if isinstance(payload.get("risks"), list) else []
    normalized = AcquiredDataset(
        dataset_id=dataset_id,
        source_identity=source,
        license=str(payload.get("license") or "unknown"),
        quality_score=float(payload.get("quality_score", 0.0) or 0.0),
        total_examples=int(payload["total_examples"]) if payload.get("total_examples") is not None else None,
        risks=risks,
        metadata={k: v for k, v in payload.items() if k not in {"dataset_id", "source_identity", "source", "license", "quality_score", "total_examples", "risks"}},
    ).to_backend_payload()
    return normalized


@dataclass(frozen=True, slots=True)
class DatasetAcquisitionApproval:
    selection_decision_id: str
    approved_candidate_ids: tuple[str, ...]
    approval_refs: tuple[str, ...]
    approved_requirements: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LocalAcquisition:
    candidate_id: str
    source_path: Path
    source_content_hash: str
    source_bytes: int
    record_format: str
    revision: str
    approval_refs: tuple[str, ...]


def acquire_approved_local_candidate(
    candidate: DatasetCandidate,
    selection: DatasetSelectionDecision,
    approval: DatasetAcquisitionApproval,
    *,
    max_bytes: int,
) -> LocalAcquisition:
    """Validate an approved local candidate without copying or executing it."""

    if selection.status != "selected" or candidate.candidate_id not in selection.selected_candidate_ids:
        raise PermissionError("candidate is not selected by the supplied selection decision")
    if approval.selection_decision_id != selection.decision_id:
        raise PermissionError("approval does not reference the supplied selection decision")
    approval_refs = tuple(sorted({str(ref).strip() for ref in approval.approval_refs if str(ref).strip()}))
    if candidate.candidate_id not in approval.approved_candidate_ids or not approval_refs:
        raise PermissionError("explicit dataset acquisition approval is missing")
    if candidate.provider_id != "local_fixture":
        raise ValueError("bounded acquisition currently supports only local_fixture candidates")
    if not candidate.artifact_ref:
        raise ValueError("local candidate has no artifact_ref")

    path = Path(candidate.artifact_ref).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"dataset artifact does not exist: {path}")
    byte_size = path.stat().st_size
    if byte_size > max_bytes:
        raise ValueError(f"dataset artifact exceeds acquisition limit: {byte_size}>{max_bytes}")
    content_hash = hash_file(path)
    declared_hash = str(candidate.provenance.get("content_hash", "")).removeprefix("sha256:")
    if declared_hash and declared_hash != content_hash:
        raise ValueError("dataset artifact hash does not match discovered provenance")
    revision = candidate.revision or f"sha256:{content_hash}"
    record_format = str(candidate.format_profile.get("record_format", "unknown"))
    if record_format not in {"jsonl", "json", "csv"}:
        raise ValueError(f"unsupported local record format: {record_format}")
    return LocalAcquisition(
        candidate_id=candidate.candidate_id,
        source_path=path,
        source_content_hash=f"sha256:{content_hash}",
        source_bytes=byte_size,
        record_format=record_format,
        revision=revision,
        approval_refs=approval_refs,
    )
