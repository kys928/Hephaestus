from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
